import os
from chromadb import PersistentClient
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_and_split_documents(directory):
    """加载并分割文档"""
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            loader = PyPDFLoader(os.path.join(directory, filename))
            docs = loader.load()
            documents.extend(docs)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len,
    )
    texts = text_splitter.split_documents(documents)
    return texts


def create_vector_db(texts):
    """创建向量数据库"""
    unique_texts = []
    seen = set()
    for doc in texts:
        # 使用文本内容作为唯一标识（可根据需要调整）
        content = doc.page_content.strip()
        if content not in seen:
            seen.add(content)
            unique_texts.append(doc)

    print(f"去重前片段数：{len(texts)}，去重后：{len(unique_texts)}")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma.from_documents(
        unique_texts,
        embeddings,
        persist_directory="./chroma_db"
    )
    return db