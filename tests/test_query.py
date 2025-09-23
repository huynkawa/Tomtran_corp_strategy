# tests/test_query.py
import sys, os
from langchain_chroma import Chroma
from src.config import make_embeddings
from src.rag_chain import rag_answer
from langchain_openai import ChatOpenAI

# Load lại vector store
db = Chroma(
    persist_directory="vector_store",
    embedding_function=make_embeddings()
)
retriever = db.as_retriever()

# Dùng GPT để trả lời
llm = ChatOpenAI(model="gpt-4o-mini")

# Ví dụ câu hỏi test
query = "phải trả ngắn hạn khác của công ty là bao nhiêu?"

print("\n=== TRẢ LỜI CHATBOT (giống app) ===\n")
answer = rag_answer(query, retriever, llm)
print(answer)
