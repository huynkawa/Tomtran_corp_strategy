# app.py (root level)
import os
import streamlit as st
import src.env  # sẽ tự nạp OPENAI_API_KEY
from src.rag_chain import rag_answer

# --- Dùng sqlite3 mặc định ---
import sqlite3

from src.prompt_loader import load_prompts, render_system_prompt
from langchain_chroma import Chroma
from src.config import make_embeddings, EMBED_MODEL

# ✅ Thêm import để tải từ Google Drive
import gdown
import zipfile

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

# === Google Drive Downloader ===
def ensure_vectorstore_from_gdrive():
    VECTOR_DIR = os.getenv("VECTOR_STORE_DIR", "vector_store")
    ZIP_PATH = "vector_store.zip"

    # 🔑 Thay ID này bằng ID file thật của bạn trên Google Drive
    GOOGLE_DRIVE_FILE_ID = "YOUR_FILE_ID_HERE"

    if not os.path.exists(VECTOR_DIR):
        st.warning("⚠️ vector_store chưa có, đang tải từ Google Drive...")
        url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
        gdown.download(url, ZIP_PATH, quiet=False)

        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(".")
        st.success("✅ Đã tải và giải nén vector_store!")

# Gọi đảm bảo vector_store có sẵn trước khi load
ensure_vectorstore_from_gdrive()

# === VectorDB Loader ===
@st.cache_resource
def get_vectordb():
    base_dir = os.getenv("VECTOR_STORE_DIR", "vector_store")
    cleaned_root = os.path.join(base_dir, "cleaned_scan_data")

    dbs = []
    if os.path.exists(cleaned_root):
        for sub in os.listdir(cleaned_root):
            sub_path = os.path.join(cleaned_root, sub)
            if os.path.isdir(sub_path):
                try:
                    db = Chroma(
                        persist_directory=sub_path,
                        embedding_function=make_embeddings()
                    )
                    dbs.append(db)
                except Exception as e:
                    print(f"⚠️ Lỗi khi load {sub_path}: {e}")

    # Nếu không có sub-db nào thì fallback
    if not dbs:
        st.warning("⚠️ Không tìm thấy sub-vectorstore nào, dùng vector_store gốc.")
        return Chroma(persist_directory=base_dir, embedding_function=make_embeddings())

    # Hợp nhất nhiều DB thành 1 retriever
    class UnionRetriever:
        def similarity_search_with_relevance_scores(self, query, k=6):
            results = []
            for db in dbs:
                try:
                    results.extend(db.similarity_search_with_relevance_scores(query, k=k))
                except Exception as e:
                    print("⚠️ Query lỗi:", e)
            results = sorted(results, key=lambda x: x[1] if x[1] else 0, reverse=True)
            return results[:k]

    return UnionRetriever()

vectordb = get_vectordb()

# === Debug diagnostics ===
with st.expander("🧪 RAG diagnostics", expanded=False):
    try:
        emb = make_embeddings()
        st.write("Embedding class:", emb.__class__.__name__)

        base_dir = os.getenv("VECTOR_STORE_DIR", "vector_store")
        cleaned_root = os.path.join(base_dir, "cleaned_scan_data")

        if os.path.exists(cleaned_root):
            st.write("Root dir:", os.path.abspath(cleaned_root))
            total = 0
            for sub in os.listdir(cleaned_root):
                sub_path = os.path.join(cleaned_root, sub)
                if os.path.isdir(sub_path):
                    try:
                        db = Chroma(
                            persist_directory=sub_path,
                            embedding_function=make_embeddings()
                        )
                        count = getattr(db, "_collection").count()
                        total += count
                        st.write(f"📂 {sub}: {count} vectors")
                    except Exception as e:
                        st.write(f"⚠️ {sub} lỗi: {e}")
            st.write(f"🔢 Tổng cộng: {total} vectors")
        else:
            st.write("Persist dir:", os.path.abspath(base_dir))
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

    # ✅ Gọi rag_answer không cần llm nữa
    result = rag_answer(
        query=latest_query,
        retriever_or_db=vectordb,
        use_fallback=fallback_general,
        threshold=MIN_RELEVANCE,
        k=K,
        debug=debug_mode,
    )

    # Decorate message theo nguồn
    assistant_id = os.getenv("ASSISTANT_ID", "unknown")

    if result["source"] == "internal":
        decorated_msg = (
            "<div style='background-color:#e8f5e9; padding:10px; border-radius:10px;'>"
            "🏛️ <b>Trả lời dựa trên kiến thức nội bộ</b> — 🤖 Assistant API</div>\n\n"
            + result["answer"]
        )
    elif result["source"] == "general":
        decorated_msg = (
            "<div style='background-color:#f5f5f5; padding:10px; border-radius:10px;'>"
            "🌐 <b>Trả lời dựa trên kiến thức tổng quan</b> — 🤖 Assistant API</div>\n\n"
            + result["answer"]
        )
    else:
        decorated_msg = "🤖 Assistant API\n\n" + result["answer"]



    st.session_state.history.append(("assistant", decorated_msg))

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
