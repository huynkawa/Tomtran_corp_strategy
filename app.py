# app.py (root level)
import os
import streamlit as st
import src.env  # sẽ tự nạp OPENAI_API_KEY
from src.rag_chain import rag_answer

# --- Dùng sqlite3 mặc định ---
import sqlite3

# --- Luôn ép dùng OpenAI client ---
from openai import OpenAI

api_key = os.getenv("OPENAI_API_KEY")
if not api_key or not api_key.strip():
    raise RuntimeError("❌ OPENAI_API_KEY chưa được cấu hình trong .env.active")

client = OpenAI(api_key=api_key)
print("[app] 🚀 OpenAI client initialized.")

from src.prompt_loader import load_prompts, render_system_prompt
from langchain_chroma import Chroma
from src.config import make_embeddings, EMBED_MODEL
from langchain_openai import ChatOpenAI

# 🚀 Cấu hình trang
st.set_page_config(page_title="TOMTRANCHATBOT", layout="wide")
st.title("TOMTRANCHATBOT")

# Load prompts, khóa luôn vào rag
cfg = load_prompts("prompts/prompts.yaml")
selected_key = "rag"

# Sidebar controls
temperature = st.sidebar.slider("Temperature", 0.0, 1.2, 0.3, 0.1)
top_p = st.sidebar.slider("Top_p", 0.1, 1.0, 1.0, 0.1)
fallback_general = st.sidebar.checkbox("Fallback GPT nếu không có tài liệu phù hợp", value=True)
K = st.sidebar.slider("Số đoạn context (k)", 1, 12, 4, 1)
MIN_RELEVANCE = st.sidebar.slider(
    "Ngưỡng điểm liên quan tối thiểu (0–1, cao = chặt)", 0.0, 1.0, 0.30, 0.05
)
debug_mode = st.sidebar.checkbox("🔧 Debug Mode", value=False)

# System prompt
system_prompt = render_system_prompt(cfg, selected_key)
with st.expander("🔧 System prompt đang dùng", expanded=False):
    st.code(system_prompt, language="markdown")

@st.cache_resource
def get_vectordb():
    vector_dir = os.getenv("VECTOR_STORE_DIR", "vector_store")
    return Chroma(persist_directory=vector_dir, embedding_function=make_embeddings())

vectordb = get_vectordb()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=temperature)

# Debug diagnostics
with st.expander("🧪 RAG diagnostics", expanded=False):
    try:
        emb = make_embeddings()
        st.write("Embedding class:", emb.__class__.__name__)
        vector_dir = os.getenv("VECTOR_STORE_DIR", "vector_store")
        st.write("Persist dir:", os.path.abspath(vector_dir))
        count = getattr(vectordb, "_collection").count()
        st.write("Vector count:", count)
    except Exception as e:
        st.warning(f"Diag error: {e}")

# --- Session state ---
if "history" not in st.session_state:
    st.session_state.history = []

# --- Input ---
user_msg = st.chat_input("Nhập câu hỏi…")
if user_msg:
    st.session_state.history.append(("user", user_msg))
    latest_query = user_msg

    result = rag_answer(
        query=latest_query,
        retriever_or_db=vectordb,   # ⚡ Ưu tiên dùng vectordb gốc
        llm=llm,
        client=client,
        use_fallback=fallback_general,
        threshold=MIN_RELEVANCE,
        k=K,
    )

    # Decorate message theo nguồn
    if result["source"] == "internal":
        decorated_msg = (
            "<div style='background-color:#e8f5e9; padding:10px; border-radius:10px;'>"
            "🏛️ <b>Trả lời dựa trên kiến thức nội bộ</b></div>\n\n"
            + result["answer"]
        )
    elif result["source"] == "general":
        decorated_msg = (
            "<div style='background-color:#f5f5f5; padding:10px; border-radius:10px;'>"
            "🌐 <b>Trả lời dựa trên kiến thức tổng quan</b></div>\n\n"
            + result["answer"]
        )
    else:
        decorated_msg = result["answer"]

    st.session_state.history.append(("assistant", decorated_msg))

    # --- Debug Mode ---
    if debug_mode:
        with st.expander("📂 Context (debug)", expanded=False):
            st.markdown(result["ctx_text"] or "—")

        if result["docs"]:
            st.write("🔎 Lấy được", len(result["docs"]), "tài liệu liên quan")
            for d in result["docs"]:
                st.text(f"- {d.metadata.get('source', 'unknown')}")

# --- Render history ---
for role, content in st.session_state.history:
    with st.chat_message(role):
        st.markdown(content, unsafe_allow_html=True)

# --- Footer ---
st.markdown(
    f"<hr><div style='text-align:center; color:gray; font-size:0.9em'>"
    f"☁️ Embedding: OpenAI – {EMBED_MODEL}</div>",
    unsafe_allow_html=True,
)
