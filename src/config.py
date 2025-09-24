from typing import Any
import os
import src.env  # đảm bảo biến môi trường đã được nạp

# --- Chroma persist dir ---
PERSIST_DIR = os.getenv("VECTOR_STORE_DIR", "vector_store")

# --- Model embedding mặc định ---
# Có thể đặt trong .env.active:
#   EMBED_MODEL=text-embedding-3-small
#   EMBED_MODEL=text-embedding-3-large
#   EMBED_MODEL=BAAI/bge-m3
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")

# --- Model re-ranker mặc định ---
# Có thể đặt trong .env.active:
#   RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
#   RERANK_MODEL=BAAI/bge-reranker-large
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")


def make_embeddings() -> Any:
    """
    Tạo embeddings theo cấu hình EMBED_MODEL.
    - Nếu EMBED_MODEL bắt đầu bằng "text-embedding" -> dùng OpenAI
    - Ngược lại -> giả định là HuggingFace model
    """
    batch = int(os.getenv("EMBED_BATCH_SIZE", "16"))

    if EMBED_MODEL.startswith("text-embedding"):
        from langchain_openai import OpenAIEmbeddings
        print(f"[config] 🚀 Using OpenAI embeddings: {EMBED_MODEL} | batch={batch}")
        return OpenAIEmbeddings(model=EMBED_MODEL, chunk_size=batch)
    else:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        print(f"[config] 🚀 Using HuggingFace embeddings: {EMBED_MODEL}")
        return HuggingFaceEmbeddings(model_name=EMBED_MODEL)
