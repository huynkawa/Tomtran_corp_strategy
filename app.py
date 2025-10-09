# app.py (root level)
import os
import streamlit as st
import src.env  # sẽ tự nạp OPENAI_API_KEY
import sqlite3
import gdown, zipfile
from src.rag_chain import rag_answer
from src.prompt_loader import load_prompts, render_system_prompt
from src.config import make_embeddings, EMBED_MODEL
from langchain_chroma import Chroma

# --- Giao diện ChatGPT style ---
from src.ui_layout import render_ui
from src.chat_saver import save_chat

# 🚀 Tạo UI và lấy các tham số
user_msg, temperature, top_p, fallback_general, K, MIN_RELEVANCE, debug_mode, show_system, show_rag = render_ui("TOMTRANCHATBOT")

# === Load Prompt cấu hình ===
cfg = load_prompts("prompts/prompts.yaml")
selected_key = "rag"

system_prompt = render_system_prompt(cfg, selected_key)
if show_system:  # ✅ chỉ hiển thị khi người dùng bật toggle
    with st.expander("🔧 System prompt đang dùng", expanded=False):
        st.code(system_prompt, language="markdown")

# === Google Drive Downloader ===
def ensure_vectorstore_from_gdrive():
    VECTOR_DIR = os.getenv("VECTOR_STORE_DIR", "vector_store")
    GOOGLE_DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1yKkKkRuYNZtSQs7biqmuMY3NGuSO9UIO?usp=sharing"

    if not os.path.exists(VECTOR_DIR):
        st.warning("⚠️ vector_store chưa có, đang tải từ Google Drive (folder)...")
        try:
            gdown.download_folder(GOOGLE_DRIVE_FOLDER_URL, output=VECTOR_DIR, quiet=False, use_cookies=False)
            st.success("✅ Đã tải folder vector_store từ Google Drive!")
        except Exception as e:
            st.error(f"❌ Lỗi khi tải vector_store: {e}")


# Gọi đảm bảo vector_store có sẵn
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

    if not dbs:
        st.warning("⚠️ Không tìm thấy sub-vectorstore nào, dùng vector_store gốc.")
        return Chroma(persist_directory=base_dir, embedding_function=make_embeddings())

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


# === RAG diagnostics ===
if show_rag:  # ✅ chỉ hiển thị khi người dùng bật toggle
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


# --- SESSION STATE cho nhiều đoạn chat ---
if "chat_histories" not in st.session_state:
    st.session_state.chat_histories = {}

# Lấy đoạn chat hiện tại (tạo mặc định nếu chưa có)
current_chat = st.session_state.get("current_chat", "Chat #1")
if current_chat not in st.session_state.chat_histories:
    st.session_state.chat_histories[current_chat] = []

# --- XỬ LÝ INPUT CHAT ---
# --- XỬ LÝ INPUT CHAT ---
if user_msg:
    # ➤ Lưu tin nhắn người dùng vào bộ nhớ
    st.session_state.chat_histories[current_chat].append(("user", user_msg))
    latest_query = user_msg

    # ➤ Gọi RAG pipeline để tìm câu trả lời
    try:
        result = rag_answer(
            query=latest_query,
            retriever_or_db=vectordb,
            use_fallback=fallback_general,
            threshold=MIN_RELEVANCE,
            k=K,
            debug=debug_mode,
        )

        # ➤ Lấy nội dung trả lời
        if isinstance(result, dict):
            answer = result.get("answer", "Không có phản hồi.")
        else:
            answer = str(result) if result else "Không có phản hồi."

        # --- Trang trí phản hồi bot ---
        decorated_msg = "🤖 Assistant API\n\n"
        if isinstance(result, dict):
            source = result.get("source", "unknown")
            if source == "internal":
                decorated_msg = answer
            elif source == "general":
                decorated_msg = (
                    "<div style='background-color:#f5f5f5; padding:10px; border-radius:10px;'>"
                    "🌐 <b>Trả lời dựa trên kiến thức tổng quan</b> — 🤖 Assistant API</div>\n\n"
                    + answer
                )
            else:
                decorated_msg += answer
        else:
            decorated_msg += str(result) if result else "Không có kết quả trả về."

        # ➤ Lưu phản hồi bot vào session (chỉ 1 lần duy nhất)
        st.session_state.chat_histories[current_chat].append(("bot", decorated_msg))

    except Exception as e:
        # ⚠️ Bắt lỗi và hiển thị ra giao diện
        st.session_state.chat_histories[current_chat].append(
            ("bot", f"⚠️ Lỗi khi xử lý truy vấn: {e}")
        )

    # 💾 Lưu hội thoại ra file JSON
    try:
        save_chat(current_chat, st.session_state.chat_histories[current_chat])
    except Exception as e:
        st.warning(f"⚠️ Không thể lưu hội thoại: {e}")

    # 🔄 Làm mới lại giao diện để hiển thị tin nhắn mới
    st.rerun()

# --- FOOTER ---
st.markdown(
    f"<hr><div style='text-align:center; color:gray; font-size:0.9em'>"
    f"☁️ Embedding: OpenAI – {EMBED_MODEL}</div>",
    unsafe_allow_html=True,
)
