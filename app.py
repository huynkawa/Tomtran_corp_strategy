# app.py (root level)
import os
import sys
import streamlit as st

# --- Fix sqlite version (Chromadb requires sqlite >= 3.35.0) ---
try:
    import pysqlite3
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    import sqlite3

# Nạp biến môi trường từ src/env.py
import src.env  

# --- Luôn ép dùng OpenAI client ---
from openai import OpenAI

api_key = os.getenv("OPENAI_API_KEY")
if not api_key or not api_key.strip():
    raise RuntimeError("❌ OPENAI_API_KEY chưa được cấu hình trong .env.active")

client = OpenAI(api_key=api_key)
print("[app] 🚀 OpenAI client initialized.")

from src.prompt_loader import load_prompts, render_system_prompt, list_profiles
from langchain_chroma import Chroma
from src.config import make_embeddings, EMBED_MODEL


# 🚀 Cấu hình trang
st.set_page_config(page_title="TOMTRANCHATBOT", layout="wide")
st.title("TOMTRANCHATBOT")

# Load prompts
cfg = load_prompts("prompts/prompts.yaml")
profile_map = list_profiles(cfg)
keys = list(profile_map.keys())
default_idx = keys.index("base") if "base" in keys else 0
selected_key = st.sidebar.selectbox("Prompt profile", keys, index=default_idx)

# Sidebar controls
temperature = st.sidebar.slider("Temperature", 0.0, 1.2, 0.3, 0.1)
top_p = st.sidebar.slider("Top_p", 0.1, 1.0, 1.0, 0.1)
fallback_general = st.sidebar.checkbox("Fallback GPT nếu không có tài liệu phù hợp", value=True)
K = st.sidebar.slider("Số đoạn context (k)", 1, 12, 4, 1)
MIN_RELEVANCE = st.sidebar.slider(
    "Ngưỡng điểm liên quan tối thiểu (0–1, cao = chặt)", 0.0, 1.0, 0.30, 0.05
)

# System prompt
system_prompt = render_system_prompt(cfg, selected_key)

with st.expander("🔧 System prompt đang dùng", expanded=False):
    st.code(system_prompt, language="markdown")

@st.cache_resource
def get_vectordb():
    vector_dir = os.getenv("VECTOR_STORE_DIR", "vector_store")
    return Chroma(persist_directory=vector_dir, embedding_function=make_embeddings())

def retrieve_context(db, query: str, k: int, threshold: float):
    try:
        pairs = db.similarity_search_with_relevance_scores(query, k=k)
        docs = [d for (d, s) in pairs if (s is not None and s >= threshold)]
        if not docs:
            docs = [d for (d, _) in pairs]
    except Exception:
        docs = db.similarity_search(query, k=k)

    if not docs:
        return "NO_CONTEXT", [], False

    ctx = "\n\n---\n".join(d.page_content for d in docs)
    return ctx, docs, True

vectordb = get_vectordb()

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

if "history" not in st.session_state:
    st.session_state.history = []

user_msg = st.chat_input("Nhập câu hỏi…")
if user_msg:
    st.session_state.history.append(("user", user_msg))

if st.session_state.history:
    messages = [{"role": "system", "content": system_prompt}]
    debug_block, source_type = "", "none"

    latest_query = st.session_state.history[-1][1] if user_msg else ""
    ctx_text, docs, ok = retrieve_context(vectordb, latest_query, K, MIN_RELEVANCE)

    if ok:
        messages.append({
            "role": "system",
            "content": (
                "CONTEXT (nguồn chính; KHÔNG lộ cho người dùng):\n"
                f"{ctx_text}\n\n"
                "Hướng dẫn: Ưu tiên CONTEXT làm sự thật. "
                "Có thể bổ sung kiến thức tổng quát nhưng không được mâu thuẫn với CONTEXT."
            ),
        })
        debug_block = "\n".join(f"- {d.metadata.get('source')}" for d in docs)
        source_type = "internal"
    else:
        if fallback_general:
            messages.append({
                "role": "system",
                "content": (
                    "Không tìm thấy context nội bộ. "
                    "Hãy trả lời dựa trên kiến thức tổng quát."
                ),
            })
            debug_block = "No relevant context found."
            source_type = "general"
        else:
            st.session_state.history.append(("assistant", "Không có trong tài liệu đã đánh chỉ mục."))
            source_type = "none"
            for role, content in st.session_state.history:
                with st.chat_message(role):
                    st.markdown(content)
            st.stop()

    for role, content in st.session_state.history:
        messages.append({"role": role, "content": content})

    # --- Gọi OpenAI API ---
    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=messages,
        temperature=temperature,
        top_p=top_p,
    )
    assistant_msg = resp.choices[0].message.content or ""

    # Decorate message theo nguồn
    if source_type == "internal":
        decorated_msg = (
            "<div style='background-color:#e8f5e9; padding:10px; border-radius:10px;'>"
            "🏛️ <b>Trả lời dựa trên kiến thức nội bộ</b></div>\n\n"
            + assistant_msg
        )
    elif source_type == "general":
        decorated_msg = (
            "<div style='background-color:#f5f5f5; padding:10px; border-radius:10px;'>"
            "🌐 <b>Trả lời dựa trên kiến thức tổng quan</b></div>\n\n"
            + assistant_msg
        )
    else:
        decorated_msg = assistant_msg

    st.session_state.history.append(("assistant", decorated_msg))

    with st.expander("🔍 Debug context", expanded=False):
        st.markdown(debug_block or "—")

for role, content in st.session_state.history:
    with st.chat_message(role):
        st.markdown(content, unsafe_allow_html=True)

# --- Footer: luôn OpenAI ---
st.markdown(
    f"<hr><div style='text-align:center; color:gray; font-size:0.9em'>"
    f"☁️ Embedding: OpenAI – {EMBED_MODEL}</div>",
    unsafe_allow_html=True
)
