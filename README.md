# 城市设计知识问答与规范解读系统

这是一个基于 **Flask + FAISS + Sentence-Transformers + SiliconFlow** 的 PDF 知识库问答系统，面向城市设计、规范解读和资料检索场景。  
项目会自动读取 `pdfs/` 目录中的 PDF 文档，提取文本后切分为知识块，构建向量索引，并通过检索增强生成（RAG）的方式回答问题。

## 功能特点

- PDF 知识库自动构建与加载
- 基于 FAISS 的语义检索
- 接入 SiliconFlow 大模型生成回答
- 支持上传新的 PDF 并重建知识库
- 支持查看系统状态
- 内置网页前端，可直接在浏览器中交互
- 可生成简单的设计说明文本

## 技术栈

- Python
- Flask
- Flask-CORS
- PyMuPDF / `fitz`
- PyPDF2
- sentence-transformers
- FAISS
- NumPy
- Requests
- Vue 2 + Axios（前端页面）

## 项目结构

```text
.
├── app.py
├── pdfs/                # 知识库 PDF 目录
├── uploads/             # 上传文件目录
├── models/              # 模型缓存目录
├── static/              # 静态资源
├── templates/           # 模板文件
├── faiss_index.bin      # 向量索引文件（运行后生成）
└── text_chunks.json     # 文本切块文件（运行后生成）
```

## 环境要求

- Python 3.9+
- 可访问外网，用于下载模型和调用 SiliconFlow 接口
- 至少准备一批可提取文本的 PDF 文件

## 安装依赖

```bash
python -m venv .venv
.venv\Scripts\activate
pip install flask flask-cors pypdf2 pymupdf sentence-transformers faiss-cpu numpy requests
```

## 使用方法

1. 将需要检索的 PDF 文件放入 `pdfs/` 目录。
2. 修改 `app.py` 中的 `SILICONFLOW_API_KEY` 为你自己的密钥。
3. 启动项目：

```bash
python app.py
```

4. 打开浏览器访问：

```text
http://127.0.0.1:5000
```

首次启动时，程序会自动：

- 加载 `moka-ai/m3e-base` 向量模型
- 读取 `pdfs/` 下的 PDF
- 生成并保存 `faiss_index.bin`
- 生成并保存 `text_chunks.json`

## 主要接口

- `GET /`：前端页面
- `POST /api/query`：知识库问答
- `POST /api/fine_tune`：上传 PDF 并更新知识库
- `POST /api/rebuild_vector_db`：手动重建向量库
- `GET /api/status`：查看系统状态
- `POST /api/explain_design`：生成设计说明文本

## 注意事项

- 如果 PDF 是扫描件或图片型文档，PyMuPDF 可能提取不到文本，需要先做 OCR
- 前端页面依赖 CDN 引入的 Vue 和 Axios，运行时需要网络可用
- `uploads/`、`faiss_index.bin`、`text_chunks.json`、`models/hf_cache/` 一般不建议直接作为公开仓库的核心内容提交


