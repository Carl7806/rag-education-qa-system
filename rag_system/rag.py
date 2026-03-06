import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate


def load_qwen_model():
    """加载 Qwen 1.5-0.5B 模型（CPU 模式）"""
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen1.5-0.5B",
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True

    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.7,
        return_full_text=False
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm


def create_rag_chain(db):
    """创建 RAG 链（使用 load_qa_chain 方式）"""
    llm = load_qwen_model()
    prompt_template = """你是一个教育领域的AI助手，回答问题要专业、简洁、准确。请基于以下上下文信息回答问题。

上下文：
{context}

问题：{question}
回答："""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # 配置检索器：使用 MMR，返回 5 个片段，候选集取 20 个
    retriever = db.as_retriever(
        search_type="mmr",  # 使用 MMR
        search_kwargs={
            "k": 5,  # 最终返回 5 个片段
            "fetch_k": 20,  # 初始取 20 个候选片段用于多样性筛选
            "lambda_mult": 0.5  # 多样性权重（0.5 平衡相关性和多样性）
        }
    )

    # 构建合并文档的问答链
    combine_chain = load_qa_chain(llm, chain_type="stuff", prompt=PROMPT)

    # 构建检索式问答链
    qa_chain = RetrievalQA(
        retriever=retriever,
        combine_documents_chain=combine_chain,
        return_source_documents=True
    )
    return qa_chain