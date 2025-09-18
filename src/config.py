from typing import Any
import os
import src.env  # NEW: đảm bảo biến môi trường đã nạp

PERSIST_DIR = os.getenv("PERSIST_DIR", "vector_store")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")


def make_embeddings() -> Any:
    use_local = os.getenv("USE_LOCAL_EMB", "0") == "1"
    batch = int(os.getenv("EMBED_BATCH_SIZE", "16"))

    if use_local:
        from langchain_huggingface import HuggingFaceEmbeddings
        model_name = os.getenv("LOCAL_EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        print(f"[config] Using LOCAL embeddings: {model_name} | batch={batch}")
        return HuggingFaceEmbeddings(model_name=model_name)

    import openai
    from langchain_openai import OpenAIEmbeddings

    openai.api_key = os.getenv("OPENAI_API_KEY")
    print(f"[config] Using OpenAI embeddings: {EMBED_MODEL} | batch={batch}")
    return OpenAIEmbeddings(model=EMBED_MODEL, chunk_size=batch)

