# rag-education-qa-system
# 教育知识问答系统（开源版）

一个基于检索增强生成（RAG） 的教育领域问答系统。它能够读取本地 PDF 文档，通过向量检索找到相关片段，并利用大语言模型生成准确、简洁的回答。系统完全本地运行，保护数据隐私。
使用Qwen 1.5-0.5B开源模型，完全无需付费API。

## 🌟 为什么选择这个项目？

- ✅ **100%开源**：使用Qwen 1.5-0.5B（Hugging Face公开模型）
- ✅ **数据合规**：所有文档来自公开教育资源（无版权风险）
- ✅ **低硬件要求**：0.5B模型可在8GB内存笔记本运行
- ✅ **技术栈匹配**：RAG、向量数据库、LLM应用（符合校招要求）

✨ 功能特点
📄 支持 PDF 文档：将你的教育资料（教材、论文、讲义）放入指定文件夹，系统自动构建知识库。

🔍 智能检索：使用 sentence-transformers/all-MiniLM-L6-v2 将文档切片转换为向量，并存储在 ChromaDB 中。

🤖 大语言模型：集成 Qwen 1.5-0.5B 模型（开源），可替换为其他 Hugging Face 模型。

🌐 Web 界面：简洁美观的 Flask 前端，支持问答交互，显示答案来源。

⚡ GPU 加速：支持 CUDA 12.1 及以上，可启用 8-bit/4-bit 量化，大幅提升推理速度。

🔄 缓存机制：自动缓存向量数据库，避免重复构建；支持语义缓存，相同问题秒回。

🛠️ 系统架构
用户输入 → Flask 前端 → 检索器 (ChromaDB) → 增强提示 → LLM (Qwen) → 答案 + 来源

📋 环境要求
Python 3.9+

操作系统：Windows

硬件：

CPU 模式：至少 4GB 内存

GPU 模式（推荐）：NVIDIA GPU，CUDA 12.1 或更高，至少 4GB 显存

## 🛠 技术栈

| 组件 | 技术 |
|------|------|
| 后端 | Python, Flask |
| RAG核心 | LangChain, ChromaDB |
| LLM | Qwen 1.5-0.5B (开源) |
| 向量模型 | all-MiniLM-L6-v2 |

## 🚀 快速启动

```bash
# 1. 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 2. 安装依赖
pip install -r requirements.txt
安装与 CUDA 12.1 兼容的最新 PyTorch 版本（2.5.1）：
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 
 --index-url https://download.pytorch.org/whl/cu121

# 3. 下载教育文档
mkdir data/education_documents
# 全部文档在Github上开源
# stanford教材
# deeplearningbook
# python-tutorial
#MIT教材
#其他相关知识的PDF文档

# 4. 运行系统
python app.py

首次启动会自动：
加载 PDF 并分割文本
生成向量并存储到 ./chroma_db
下载 Qwen 模型

看到如下输出后，打开浏览器访问 http://127.0.0.1:5000：
* Running on http://127.0.0.1:5000

#5 使用

在输入框中输入问题，点击“提问”，稍等片刻即可获得答案及参考来源。
```
⚙️ 配置说明

修改模型

替换 embedding 模型：编辑 vector_db.py 中的 HuggingFaceEmbeddings(model_name="your-embedding-model")

替换 LLM：编辑 rag.py 中的 AutoModelForCausalLM.from_pretrained("your-llm")

调整检索参数
在 rag.py 的 create_rag_chain 中修改检索器配置：
```
retriever = db.as_retriever(
    search_type="mmr",  # 使用最大边际相关性检索，增加多样性
    search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.5}
)
```

修改文本分割参数
编辑 vector_db.py 中的 RecursiveCharacterTextSplitter：
````
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,      # 每个片段字符数
    chunk_overlap=100,    # 重叠字符数
    length_function=len,
)
````

📁 项目结构
````
.
├── app.py                  # Flask 主应用
├── requirements.txt        # 依赖列表
├── data/
│   └── education_documents/ # 存放 PDF 文档（需手动添加）
├── rag_system/
│   ├── vector_db.py        # 向量数据库构建与加载
│   └── rag.py              # RAG 核心逻辑（模型加载、问答链）
├── templates/
│   └── index.html          # Web 界面模板
└── chroma_db/              # 自动生成的向量数据库（首次运行后出现）
````

运行截图

<img width="1124" height="688" alt="屏幕截图 2026-03-06 232635" src="https://github.com/user-attachments/assets/b3ce1e1e-f2b0-452f-877a-3b547da6185b" />

<img width="1128" height="606" alt="屏幕截图 2026-03-06 232652" src="https://github.com/user-attachments/assets/fb6117a9-45b2-4d03-af13-2a8df247dbbe" />


❓ 常见问题
1. 启动时提示网络错误，无法下载模型
设置 Hugging Face 镜像：set HF_ENDPOINT=https://hf-mirror.com（Windows）

或手动下载模型后修改代码中的路径为本地路径。

2. 回答中包含了上下文（prompt）
在 pipeline 中添加 return_full_text=False（已在代码中默认设置）。

3. 响应速度慢
启用 GPU 加速（见配置说明）。

减少检索返回的片段数（k 值）和生成的最大 token 数（max_new_tokens）。

添加语义缓存（已在 app.py 示例中提供）。

4. 来源片段重复
在构建向量库时添加去重逻辑（参考 vector_db.py 中的注释）。

使用 MMR 检索（search_type="mmr"）增加多样性。

📝 后续开发

添加更多文件格式：修改 vector_db.py 中的 load_and_split_documents，增加对 .txt、.docx 等的支持。

更换前端：编辑 templates/index.html 调整样式和交互。

添加用户认证：扩展 Flask 应用，增加登录功能。

📄 许可证
本项目采用 MIT 许可证。详见 LICENSE 文件。

🙏 致谢

·LangChain

·Hugging Face

·Qwen

·ChromaDB
