# app.py (root level)
import os
import sys
import streamlit as st

# --- Fix sqlite version (Chromadb requires sqlite >= 3.35.0) ---
try:
    import pysqlite3
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ModuleNotFoundError:
    # Nếu local Windows không có pysqlite3 thì vẫn dùng sqlite3 mặc định
    pass

# nạp biến môi trường từ src/env.py
import src.env  

# --- Khởi tạo OpenAI client an toàn ---
USE_CLIENT = False
client = None

try:
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key and api_key.strip():
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        USE_CLIENT = True
        print("[app] ✅ OpenAI client initialized.")
    else:
        print("[app] ⚠️ Không có OPENAI_API_KEY → chỉ chạy Local HuggingFace.")
except ImportError:
    pass

from src.prompt_loader import load_prompts, render_system_prompt, list_profiles
from langchain_chroma import Chroma
from src.config import make_embeddings, EMBED_MODEL
