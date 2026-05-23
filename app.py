from flask import Flask, request, jsonify, render_template_string, make_response
from flask_cors import CORS
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import json
from datetime import datetime
import PyPDF2  # 或使用 fitz (PyMuPDF)
import fitz  # PyMuPDF，推荐
import re
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import requests  # 用于调用硅基流动 API

app = Flask(__name__)
CORS(app)

# ================== 配置 ==================
UPLOAD_FOLDER = 'uploads'
KNOWLEDGE_DIR = 'pdfs'  # 存放PDF知识库
VECTOR_DB_PATH = 'faiss_index.bin'
CHUNKS_PATH = 'text_chunks.json'

# 硅基流动 API 配置（请替换为你的 API Key）
SILICONFLOW_API_KEY = "sk-snsrtwxzrdvkwvcodkfoetvrludxtvtfgseeltxdssozezxu"  # 替换为你的密钥
SILICONFLOW_MODEL = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"  # 支持多种模型，如 Qwen、DeepSeek 等

# 创建目录
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(KNOWLEDGE_DIR, exist_ok=True)

# 加载嵌入模型（中文推荐使用 m3e 或 bge）
embedding_model = SentenceTransformer('moka-ai/m3e-base', cache_folder='models/hf_cache')

# 存储对话历史
conversation_history = []

# 全局变量：文档块与向量索引
text_chunks = []
index = None


# ================== PDF 处理与向量数据库构建 ==================

def extract_text_from_pdf(pdf_path):
    """使用 PyMuPDF 提取 PDF 文本"""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        print(f"提取PDF文本失败 {pdf_path}: {str(e)}")
        return ""


def split_text(text, chunk_size=300, overlap=50):
    """将文本切分为块"""
    if not text or not text.strip():
        return []

    text = re.sub(r'\s+', ' ', text)  # 压缩空白
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end > len(text):
            end = len(text)
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def build_vector_db():
    """从 knowledge_pdfs 目录中所有 PDF 构建 FAISS 向量库"""
    global text_chunks, index

    all_chunks = []
    pdf_files = [f for f in os.listdir(KNOWLEDGE_DIR) if f.lower().endswith('.pdf')]

    if not pdf_files:
        print(" 未在 knowledge_pdfs 目录中找到任何 PDF 文件")
        return False

    for pdf_file in pdf_files:
        pdf_path = os.path.join(KNOWLEDGE_DIR, pdf_file)
        print(f"正在处理 PDF: {pdf_file}")
        text = extract_text_from_pdf(pdf_path)
        if not text:
            print(f" 警告: {pdf_file} 未提取到文本，跳过")
            continue

        chunks = split_text(text, chunk_size=300, overlap=50)
        # 添加元数据
        all_chunks.extend([{
            "text": chunk,
            "source": f"{pdf_file}#P{idx // 10 + 1}"  # 简单分页
        } for idx, chunk in enumerate(chunks)])

    if not all_chunks:
        print("未提取到任何文本块")
        return False

    # 保存文本块
    text_chunks = all_chunks
    with open(CHUNKS_PATH, 'w', encoding='utf-8') as f:
        json.dump(text_chunks, f, ensure_ascii=False, indent=2)

    # 生成嵌入向量
    print("正在生成嵌入向量...")
    try:
        embeddings = embedding_model.encode([chunk['text'] for chunk in text_chunks], convert_to_numpy=True)
        dimension = embeddings.shape[1]

        # 构建 FAISS 索引
        index = faiss.IndexFlatL2(dimension)
        faiss.normalize_L2(embeddings)  # 归一化用于内积搜索
        index.add(embeddings)

        # 保存索引
        faiss.write_index(index, VECTOR_DB_PATH)
        print(f"向量数据库构建完成，共 {len(text_chunks)} 个文本块")
        return True
    except Exception as e:
        print(f"构建向量数据库失败: {str(e)}")
        return False


def load_vector_db():
    """加载已构建的向量数据库"""
    global text_chunks, index

    # 重置为 None 确保类型正确
    index = None
    text_chunks = []

    if os.path.exists(VECTOR_DB_PATH) and os.path.exists(CHUNKS_PATH):
        try:
            print("加载已有向量数据库...")
            # 明确重新赋值，避免函数引用
            loaded_index = faiss.read_index(VECTOR_DB_PATH)
            index = loaded_index  # 直接赋值

            with open(CHUNKS_PATH, 'r', encoding='utf-8') as f:
                text_chunks = json.load(f)
            print(f"已加载 {len(text_chunks)} 个文本块")

            # 验证索引类型
            if hasattr(index, 'search'):
                print("[SUCCESS] FAISS 索引加载成功，search 方法可用")
                return True
            else:
                print("[ERROR] FAISS 索引加载失败，search 方法不可用")
                return False

        except Exception as e:
            print(f"加载向量数据库失败: {str(e)}")
            # 如果加载失败，尝试重新构建
            print("尝试重新构建向量数据库...")
            return build_vector_db()
    else:
        print("向量数据库不存在，开始构建...")
        return build_vector_db()


# 初始化时加载向量库
print("初始化向量数据库...")
vector_db_loaded = load_vector_db()
if vector_db_loaded:
    print("[SUCCESS] 向量数据库初始化成功")
else:
    print("[ERROR] 向量数据库初始化失败")


# ================== 调用硅基流动 API ==================
def call_siliconflow(prompt, context=""):
    """调用硅基流动 API 生成回答"""
    url = "https://api.siliconflow.cn/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
        "Content-Type": "application/json"
    }
    full_prompt = f"{context}\n\n问题：{prompt}\n请基于以上信息给出专业、准确的回答。"
    payload = {
        "model": SILICONFLOW_MODEL,
        "messages": [
            {"role": "user", "content": full_prompt}
        ],
        "max_tokens": 512,
        "temperature": 0.7,
        "stream": False
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            data = response.json()
            return data['choices'][0]['message']['content']
        else:
            return f"调用硅基流动失败：{response.status_code} {response.text}"
    except Exception as e:
        return f"请求异常：{str(e)}"


# ================== API 接口 ==================

@app.route('/')
def index():
    response = make_response(HTML_TEMPLATE)
    response.headers['Content-Type'] = 'text/html; charset=utf-8'
    return response


@app.route('/api/query', methods=['POST'])
def query():
    """RAG问答接口：检索+生成"""
    global index, text_chunks

    data = request.json
    question = data.get('question', '').strip()
    if not question:
        return jsonify({"error": "问题不能为空"}), 400

    # 详细检查向量数据库状态
    if index is None:
        print("错误: index 为 None")
        return jsonify({"error": "知识库索引未初始化"}), 500

    if not hasattr(index, 'search'):
        print(f"错误: index 类型为 {type(index)}，没有 search 方法")
        # 尝试重新加载
        print("尝试重新加载向量数据库...")
        if load_vector_db():
            if index is None or not hasattr(index, 'search'):
                return jsonify({"error": "知识库索引损坏，请重新构建"}), 500
        else:
            return jsonify({"error": "知识库加载失败"}), 500

    if not text_chunks:
        return jsonify({"error": "知识库文本块为空"}), 500

    try:
        # 1. 向量化问题
        query_embedding = embedding_model.encode([question])
        faiss.normalize_L2(query_embedding)

        # 2. 检索最相似的 top-k 文本块
        k = min(3, len(text_chunks))  # 确保 k 不超过文本块数量
        distances, indices = index.search(query_embedding, k)

        # 处理搜索结果
        relevant_docs = []
        for i in indices[0]:
            if i < len(text_chunks) and i >= 0:  # 确保索引有效
                relevant_docs.append(text_chunks[i])

        if not relevant_docs:
            return jsonify({"error": "未找到相关文档内容"}), 404

        # 3. 构建上下文
        context = "\n\n".join([doc['text'] for doc in relevant_docs])

        # 4. 调用硅基流动生成回答
        answer = call_siliconflow(question, context)

        # 5. 记录对话
        conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer,
            "sources": [doc['source'] for doc in relevant_docs]
        })

        return jsonify({
            "answer": answer,
            "sources": [doc['source'] for doc in relevant_docs],
            "relevant_docs": relevant_docs
        })

    except Exception as e:
        print(f"查询过程中出错: {str(e)}")
        return jsonify({"error": f"查询失败: {str(e)}"}), 500


@app.route('/api/explain_design', methods=['POST'])
def explain_design():
    """生成设计说明书（可扩展为调用大模型）"""
    data = request.json
    requirements = data.get('requirements', '')

    # 可改为调用硅基流动生成更专业文档
    document = generate_design_document(requirements)
    return jsonify({
        "document": document,
        "generated_at": datetime.now().isoformat()
    })


@app.route('/api/fine_tune', methods=['POST'])
def fine_tune():
    """上传新PDF并更新知识库"""
    global index, text_chunks

    if 'file' not in request.files:
        return jsonify({"error": "没有上传文件"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "未选择文件"}), 400

    if not file.filename.lower().endswith('.pdf'):
        return jsonify({"error": "只支持 PDF 文件"}), 400

    try:
        # 保存文件到知识库目录
        filepath = os.path.join(KNOWLEDGE_DIR, file.filename)
        file.save(filepath)

        # 重新构建向量数据库
        success = build_vector_db()

        if success:
            return jsonify({
                "message": "PDF上传成功并已更新知识库",
                "filename": file.filename,
                "status": "success"
            })
        else:
            return jsonify({"error": "PDF上传成功但知识库更新失败"}), 500

    except Exception as e:
        return jsonify({"error": f"上传失败: {str(e)}"}), 500


@app.route('/api/status', methods=['GET'])
def status():
    """系统状态"""
    global index, text_chunks

    vector_db_status = "loaded" if index is not None and hasattr(index, 'search') else "not loaded"
    index_type = str(type(index)) if index is not None else "None"

    return jsonify({
        "status": "online",
        "model": SILICONFLOW_MODEL,
        "knowledge_base_size": len(text_chunks) if text_chunks else 0,
        "vector_db_status": vector_db_status,
        "index_type": index_type,
        "last_updated": datetime.now().isoformat(),
        "vector_db": "FAISS + m3e-base"
    })


@app.route('/api/rebuild_vector_db', methods=['POST'])
def rebuild_vector_db():
    """手动重新构建向量数据库"""
    global index, text_chunks

    try:
        success = build_vector_db()
        if success:
            return jsonify({
                "message": "向量数据库重建成功",
                "chunks_count": len(text_chunks),
                "status": "success"
            })
        else:
            return jsonify({"error": "向量数据库重建失败"}), 500
    except Exception as e:
        return jsonify({"error": f"重建失败: {str(e)}"}), 500


# ================== 辅助函数 ==================
def generate_design_document(requirements):
    """模拟生成设计说明书（可替换为调用大模型）"""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"""设计计算说明书

生成时间：{current_time}

设计需求：{requirements}

内容摘要：
根据您提供的设计需求，系统已结合城市排水、防涝等相关规范进行分析。

建议方案：
1. 采用合理的管道坡度设计，确保排水通畅；
2. 设置雨水调蓄池以应对短时强降雨；
3. 关键节点部署智能积水监测设备。

详细计算需结合具体地形与降雨数据。

—— 本内容由 AI 自动生成，仅供参考。"""


# ================== 前端 HTML 模板（保持不变）==================
HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>城市设计知识辅助与规范解读系统</title>
    <script src="https://cdn.bootcdn.net/ajax/libs/vue/2.6.14/vue.min.js"></script>
    <script src="https://cdn.bootcdn.net/ajax/libs/axios/0.21.1/axios.min.js"></script>
    <style>
        /* 你的 style 保持不变 */
        * { margin: 0; padding: 0; box-sizing: border-box; font-family: 'Microsoft YaHei', sans-serif; }
        body { background-color: #f5f7fa; color: #333; line-height: 1.6; }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        header { background: linear-gradient(135deg, #1e5799 0%, #207cca 100%); color: white; padding: 20px 0; border-radius: 8px 8px 0 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px; }
        .header-content { display: flex; justify-content: space-between; align-items: center; padding: 0 20px; }
        .logo { font-size: 24px; font-weight: bold; }
        .nav-tabs { display: flex; background-color: #fff; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }
        .tab { padding: 15px 25px; cursor: pointer; transition: all 0.3s ease; font-weight: 500; }
        .tab.active { background-color: #1e5799; color: white; }
        .tab:hover:not(.active) { background-color: #f0f5ff; }
        .content-area { display: flex; gap: 20px; }
        .main-content { flex: 3; background-color: white; border-radius: 8px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .sidebar { flex: 1; background-color: white; border-radius: 8px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .chat-container { height: 500px; display: flex; flex-direction: column; }
        .chat-messages { flex: 1; overflow-y: auto; padding: 15px; border: 1px solid #e1e4e8; border-radius: 8px; margin-bottom: 15px; background-color: #fafbfc; }
        .message { margin-bottom: 15px; padding: 10px 15px; border-radius: 8px; max-width: 80%; }
        .user-message { background-color: #e3f2fd; margin-left: auto; border: 1px solid #bbdefb; }
        .system-message { background-color: #f5f5f5; margin-right: auto; border: 1px solid #e0e0e0; }
        .message-header { font-weight: bold; margin-bottom: 5px; font-size: 14px; }
        .message-content { white-space: pre-wrap; }
        .message-sources { margin-top: 5px; font-size: 12px; color: #666; }
        .chat-input { display: flex; gap: 10px; }
        .chat-input textarea { flex: 1; padding: 12px; border: 1px solid #e1e4e8; border-radius: 8px; resize: none; height: 80px; }
        .chat-input button { padding: 0 20px; background-color: #1e5799; color: white; border: none; border-radius: 8px; cursor: pointer; }
        .chat-input button:hover { background-color: #16437e; }
        .document-preview { margin-top: 20px; border: 1px solid #e1e4e8; border-radius: 8px; padding: 15px; background-color: #fafbfc; max-height: 300px; overflow-y: auto; }
        .document-preview h3 { margin-bottom: 10px; color: #1e5799; }
        .document-content { white-space: pre-wrap; font-family: 'Courier New', monospace; font-size: 14px; }
        .action-buttons { display: flex; gap: 10px; margin-top: 15px; }
        .action-buttons button { padding: 8px 15px; background-color: #1e5799; color: white; border: none; border-radius: 4px; cursor: pointer; }
        .action-buttons button:hover { background-color: #16437e; }
        .sidebar-section h3 { margin-bottom: 10px; color: #1e5799; border-bottom: 1px solid #e1e4e8; padding-bottom: 5px; }
        .knowledge-list { list-style-type: none; }
        .knowledge-list li { padding: 8px 0; border-bottom: 1px solid #f0f0f0; }
        .status-indicator { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 8px; }
        .status-online { background-color: #4caf50; }
        .status-offline { background-color: #f44336; }
        .loading { text-align: center; padding: 10px; color: #666; }
        .citation { background-color: #fff8e1; padding: 5px 10px; border-radius: 4px; margin-top: 5px; font-size: 12px; border-left: 3px solid #ffd54f; }
        footer { text-align: center; margin-top: 30px; padding: 15px; color: #666; font-size: 14px; }
    </style>
</head>
<body>
    <div id="app">
        <header>
            <div class="header-content">
                <div class="logo">城市设计知识辅助与规范解读系统</div>
                <div class="system-status">
                    <span class="status-indicator" :class="systemStatus ? 'status-online' : 'status-offline'"></span>
                    {{ systemStatus ? '系统在线' : '系统离线' }}
                </div>
            </div>
        </header>

        <div class="container">
            <div class="nav-tabs">
                <div class="tab" :class="{active: activeTab === 'qa'}" @click="activeTab = 'qa'">智能问答</div>
                <div class="tab" :class="{active: activeTab === 'doc'}" @click="activeTab = 'doc'">文档生成</div>
                <div class="tab" :class="{active: activeTab === 'fine-tune'}" @click="activeTab = 'fine-tune'">模型微调</div>
                <div class="tab" :class="{active: activeTab === 'debug'}" @click="activeTab = 'debug'">调试</div>
            </div>

            <div class="content-area">
                <div class="main-content">
                    <!-- 智能问答 -->
                    <div v-if="activeTab === 'qa'" class="qa-panel">
                        <h2>规范智能问答</h2>
                        <p>输入您的问题，系统将基于规范库提供准确解答</p>
                        <div class="chat-container">
                            <div class="chat-messages">
                                <div v-for="(message, index) in chatMessages" :key="index" class="message" :class="message.type === 'user' ? 'user-message' : 'system-message'">
                                    <div class="message-header">{{ message.type === 'user' ? '您' : '系统' }}</div>
                                    <div class="message-content">{{ message.content }}</div>
                                    <div v-if="message.sources && message.sources.length" class="message-sources">
                                        <strong>参考来源:</strong>
                                        <div v-for="(source, sIndex) in message.sources" :key="sIndex" class="citation">{{ source }}</div>
                                    </div>
                                </div>
                                <div v-if="loading" class="loading">系统正在思考中...</div>
                            </div>
                            <div class="chat-input">
                                <textarea v-model="userQuestion" placeholder="请输入您的问题..." @keydown.enter.exact.prevent="sendQuestion"></textarea>
                                <button @click="sendQuestion">发送</button>
                            </div>
                        </div>
                    </div>

                    <!-- 文档生成 -->
                    <div v-if="activeTab === 'doc'" class="doc-panel">
                        <h2>设计文档生成</h2>
                        <p>基于规范自动生成设计计算说明书初稿</p>
                        <div class="document-input">
                            <textarea v-model="designRequirements" placeholder="请输入设计需求..." rows="5"></textarea>
                            <div class="action-buttons">
                                <button @click="generateDocument">生成文档</button>
                                <button @click="downloadDocument" :disabled="!generatedDocument">下载文档</button>
                            </div>
                        </div>
                        <div v-if="generatedDocument" class="document-preview">
                            <h3>生成的设计计算说明书</h3>
                            <div class="document-content">{{ generatedDocument }}</div>
                        </div>
                    </div>

                    <!-- 模型微调（上传PDF） -->
                    <div v-if="activeTab === 'fine-tune'" class="fine-tune-panel">
                        <h2>模型微调（上传PDF）</h2>
                        <p>上传新的PDF规范文件以扩展知识库</p>
                        <div class="upload-area">
                            <input type="file" id="file-upload" @change="handleFileUpload" accept=".pdf">
                            <label for="file-upload" class="upload-label">选择PDF文件</label>
                            <span v-if="uploadedFile" class="file-name">{{ uploadedFile.name }}</span>
                        </div>
                        <div class="action-buttons">
                            <button @click="startFineTuning" :disabled="!uploadedFile">上传并更新知识库</button>
                            <button @click="rebuildVectorDb">重建向量数据库</button>
                        </div>
                        <div v-if="fineTuneStatus" class="fine-tune-status">
                            <h3>状态</h3>
                            <div>{{ fineTuneStatus }}</div>
                        </div>
                    </div>

                    <!-- 调试页面 -->
                    <div v-if="activeTab === 'debug'" class="debug-panel">
                        <h2>系统调试</h2>
                        <div class="action-buttons">
                            <button @click="checkVectorDb">检查向量数据库状态</button>
                            <button @click="rebuildVectorDb">重建向量数据库</button>
                        </div>
                        <div v-if="debugInfo" class="document-preview">
                            <h3>调试信息</h3>
                            <div class="document-content">{{ debugInfo }}</div>
                        </div>
                    </div>
                </div>

                <div class="sidebar">
                    <div class="sidebar-section">
                        <h3>知识库状态</h3>
                        <ul class="knowledge-list">
                            <li v-for="file in pdfFiles" :key="file">
                                <span class="status-indicator status-online"></span> {{ file }}
                            </li>
                        </ul>
                    </div>
                    <div class="sidebar-section">
                        <h3>系统信息</h3>
                        <p>模型: {{ modelInfo.name }}</p>
                        <p>版本: {{ modelInfo.version }}</p>
                        <p>语料库: {{ modelInfo.corpusSize }} 条</p>
                        <p>最后更新: {{ modelInfo.lastUpdate }}</p>
                    </div>
                </div>
            </div>
        </div>

        <footer>
            <p>基于大模型与RAG的城市设计知识辅助与规范解读系统 &copy; 2025</p>
        </footer>
    </div>

    <script>
        new Vue({
            el: '#app',
            data: {
                activeTab: 'qa',
                systemStatus: true,
                userQuestion: '',
                chatMessages: [],
                loading: false,
                designRequirements: '',
                generatedDocument: '',
                uploadedFile: null,
                fineTuneStatus: '',
                debugInfo: '',
                modelInfo: {
                    name: 'deepseek-ai/DeepSeek-R1-0528-Qwen3-8B',
                    version: '1.0',
                    corpusSize: 0,
                    indexStatus: '未知',
                    lastUpdate: new Date().toLocaleDateString()
                },
                pdfFiles: []
            },
            mounted() {
                this.fetchStatus();
            },
            methods: {
                async fetchStatus() {
                    try {
                        const res = await axios.get('/api/status');
                        this.modelInfo.corpusSize = res.data.knowledge_base_size;
                        this.modelInfo.indexStatus = res.data.vector_db_status;
                        this.modelInfo.lastUpdate = new Date().toLocaleDateString();
                        // 模拟列出PDF（实际可扩展）
                        this.pdfFiles = ['GBT+34173-2017.pdf', 'GBT+39195-2020.pdf'];
                    } catch (e) {
                        console.error(e);
                    }
                },
                async sendQuestion() {
                    const q = this.userQuestion.trim();
                    if (!q) return;
                    this.chatMessages.push({ type: 'user', content: q });
                    this.userQuestion = '';
                    this.loading = true;
                    try {
                        const res = await axios.post('/api/query', { question: q });
                        this.chatMessages.push({
                            type: 'system',
                            content: res.data.answer,
                            sources: res.data.sources
                        });
                    } catch (e) {
                        this.chatMessages.push({
                            type: 'system',
                            content: '请求失败: ' + (e.response?.data?.error || e.message)
                        });
                    } finally {
                        this.loading = false;
                    }
                },
                generateDocument() {
                    axios.post('/api/explain_design', { requirements: this.designRequirements })
                        .then(res => this.generatedDocument = res.data.document);
                },
                downloadDocument() {
                    const blob = new Blob([this.generatedDocument], { type: 'text/plain' });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = '设计说明书.txt';
                    a.click();
                    URL.revokeObjectURL(url);
                },
                handleFileUpload(e) {
                    this.uploadedFile = e.target.files[0];
                },
                async startFineTuning() {
                    if (!this.uploadedFile) return;
                    const formData = new FormData();
                    formData.append('file', this.uploadedFile);
                    this.fineTuneStatus = '上传中...';
                    try {
                        const res = await axios.post('/api/fine_tune', formData, {
                            headers: { 'Content-Type': 'multipart/form-data' }
                        });
                        this.fineTuneStatus = res.data.message;
                        setTimeout(() => this.fineTuneStatus = '', 3000);
                    } catch (e) {
                        this.fineTuneStatus = '上传失败: ' + e.message;
                    }
                },
                async rebuildVectorDb() {
                    try {
                        const res = await axios.post('/api/rebuild_vector_db');
                        this.fineTuneStatus = res.data.message;
                        this.fetchStatus();
                    } catch (e) {
                        this.fineTuneStatus = '重建失败: ' + e.message;
                    }
                },
                async checkVectorDb() {
                    try {
                        const res = await axios.get('/api/status');
                        this.debugInfo = JSON.stringify(res.data, null, 2);
                    } catch (e) {
                        this.debugInfo = '检查失败: ' + e.message;
                    }
                }
            }
        });
    </script>
</body>
</html>
'''

# ================== 启动服务 ==================
if __name__ == '__main__':
    print("🚀 启动服务中...")
    app.run(debug=True, host='0.0.0.0', port=5000)