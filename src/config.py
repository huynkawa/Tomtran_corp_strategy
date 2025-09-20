from typing import Any
import os
import src.env  # đảm bảo biến môi trường đã được nạp

# --- Chroma persist dir ---
PERSIST_DIR = os.getenv("VECTOR_STORE_DIR", "vector_store")

# --- Model embedding mặc định (OpenAI) ---
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")


def make_embeddings() -> Any:
    """
    Luôn dùng OpenAI Embeddings (không còn fallback HuggingFace).
    """
    from langchain_openai import OpenAIEmbeddings

    batch = int(os.getenv("EMBED_BATCH_SIZE", "16"))
    print(f"[config] 🚀 Forcing OpenAI embeddings: {EMBED_MODEL} | batch={batch}")
    return OpenAIEmbeddings(model=EMBED_MODEL, chunk_size=batch)
