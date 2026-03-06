import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from flask import Flask, render_template, request, jsonify
from rag_system.vector_db import load_and_split_documents, create_vector_db
from rag_system.rag import create_rag_chain
from functools import lru_cache
from collections import defaultdict

app = Flask(__name__)

# 初始化系统（首次运行时自动创建向量库）
db = create_vector_db(load_and_split_documents("data/education_documents"))
qa_chain = create_rag_chain(db)

@app.route('/')
def home():
    return render_template('index.html')

@lru_cache(maxsize=100)
def get_answer(question):
    result = qa_chain.invoke({"query": question})
    return result['result'], result.get('source_documents', [])


@app.route('/ask', methods=['POST'])
def ask():
    question = request.form['question']
    result = qa_chain.invoke({"query": question})  # 传入字典
    answer = result['result']
    sources = result.get('source_documents', [])
    unique_sources = []
    seen_files = set()
    for doc in sources:
        file_name = doc.metadata.get('source', '').split('/')[-1]
        if file_name not in seen_files:
            seen_files.add(file_name)
            unique_sources.append(doc)


    return render_template('index.html',
                          question=question,
                          answer=answer,
                          sources=sources)

if __name__ == '__main__':
    app.run(debug=True)