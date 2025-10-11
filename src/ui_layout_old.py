# src/ui_streamlit_layout.py
# -*- coding: utf-8 -*-
"""
Giao diện giống ChatGPT thật — full screen, chat box dính đáy
Có logo động (MP4) loop vô hạn ở đầu trang.
"""

import os
import streamlit as st
from io import BytesIO
from docx import Document
from PyPDF2 import PdfReader
from src.chat_saver import save_chat


# --- đọc nội dung file (txt/pdf/docx) ---
def read_text_from_file(uploaded_file):
    try:
        if uploaded_file.name.endswith(".txt"):
            return uploaded_file.read().decode("utf-8", errors="ignore")

        elif uploaded_file.name.endswith(".pdf"):
            reader = PdfReader(uploaded_file)
            text = ""
            for page in reader.pages[:3]:
                text += page.extract_text() + "\n"
            return text or "[Không đọc được nội dung PDF]"

        elif uploaded_file.name.endswith(".docx"):
            doc = Document(uploaded_file)
            text = "\n".join([p.text for p in doc.paragraphs[:30]])
            return text or "[Không đọc được nội dung DOCX]"
        else:
            return "[Định dạng không hỗ trợ]"
    except Exception as e:
        return f"[Lỗi đọc file: {e}]"


def render_ui(title="TOMTRANCHATBOT"):
    st.set_page_config(page_title=title, page_icon="💬", layout="wide")

    # ========== CSS ================
    st.markdown("""
    <style>
    body { background-color: #0E1117; color: white; }
    [data-testid="stSidebar"] { background-color: #1C1F26; border-right: 1px solid #2E2E2E; }

    /* 🧾 Tiêu đề h1 */
    h1 {
        text-align: center;
        font-weight: 700;
        color: white;
        margin-bottom: 0.2rem;
    }

    /* 💬 Khung chat chính */
    .chat-container {
        display: flex;
        flex-direction: column;
        justify-content: flex-end;
        height: 60vh;                  /* 🔹 Giảm chiều cao để tránh che chữ */
        overflow-y: auto;
        padding: 0.8rem 1.5rem;        /* 🔹 Giảm padding */
        background-color: #0E1117;
        border-radius: 10px;
        margin-top: -4rem;           /* 🔹 Kéo gần lên tiêu đề hơn */
    }


    .user-message {
        background-color: #2A2D34;
        padding: 12px 16px;
        border-radius: 12px;
        margin: 8px 0;
        text-align: right;
        max-width: 75%;
        align-self: flex-end;
    }

    .bot-message {
        background-color: #1E1F25;
        padding: 12px 16px;
        border-radius: 12px;
        margin: 8px 0;
        text-align: left;
        max-width: 75%;
        align-self: flex-start;
    }

    .file-message {
        background-color: #1E252F;
        border-radius: 8px;
        padding: 10px;
        margin: 6px 0;
        font-size: 0.9em;
        color: #ccc;
    }

    .stChatInputContainer textarea {
        border-radius: 8px !important;
        background-color: #1C1F26 !important;
        color: white !important;
        min-height: 70px !important;
        font-size: 1.05rem !important;
        padding-top: 1rem !important;
        line-height: 1.4 !important;
    }

    /* 🔹📱 Responsive cho mobile */
    @media (max-width: 900px) {
        [data-testid="stSidebar"] { display: none !important; }
        .chat-container {
            height: 85vh !important;
            padding: 0.5rem 1rem !important;
        }
        .stChatInputContainer {
            left: 1rem !important;
            right: 1rem !important;
            bottom: 0.5rem !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)

    # --- Responsive: Cho phép bật/tắt sidebar thủ công ---
    st.markdown("""
        <style>
        @media (max-width: 768px) {
            [data-testid="stSidebar"] { display: none; }
            .open-sidebar {
                position: fixed; top: 10px; left: 10px;
                background-color: #1C1F26; color: white;
                border-radius: 6px; padding: 6px 10px;
                font-size: 20px; cursor: pointer;
                z-index: 9999; box-shadow: 0 2px 6px rgba(0,0,0,0.3);
                transition: all 0.2s ease;
            }
            .open-sidebar:hover { background-color: #292d36; }
        }
        </style>
        <div class="open-sidebar" onclick="window.parent.postMessage(
            {type: 'streamlit:setSidebarState', value: 'expanded'}, '*'
        )">☰</div>
    """, unsafe_allow_html=True)

    # ========== SIDEBAR ============
    with st.sidebar:
        st.markdown(f"## 💬 {title}")

        if "chat_sessions" not in st.session_state:
            st.session_state.chat_sessions = ["Đoạn Chat 1"]

        if st.button("➕ Tạo đoạn chat mới"):
            from datetime import datetime
            new_name = f"Chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            st.session_state.chat_sessions.append(new_name)
            st.session_state.current_chat = new_name
            st.session_state.chat_histories[new_name] = []
            save_chat(new_name, [])
            st.rerun()

        selected_chat = st.selectbox(
            "Chọn đoạn chat:",
            st.session_state.chat_sessions,
            index=len(st.session_state.chat_sessions) - 1
        )
        st.session_state.current_chat = selected_chat

        st.divider()

        show_debug_ui = os.getenv("SHOW_DEBUG_UI", "true").lower() == "true"

        if show_debug_ui:
            st.markdown("### 👁️ Hiển thị các phần")
            show_settings = st.checkbox("⚙️ Cài đặt mô hình", value=True)
            show_system = st.checkbox("📜 System prompt", value=False)
            show_rag = st.checkbox("🧠 RAG diagnostics", value=False)
        else:
            show_settings = show_system = show_rag = False

        temperature = 0.3
        top_p = 1.0
        fallback_general = True
        K = 4
        MIN_RELEVANCE = 0.3
        debug_mode = False

    # ========== MAIN CHAT AREA ============

    # --- 🎥 Logo động nhỏ gọn (MP4 loop vô hạn) ---
    # --- 🎥 Logo động nhỏ gọn (MP4 loop vô hạn, hiển thị chính xác) ---
    from pathlib import Path

    logo_path = Path("assets/robot_logo.mp4")
    if logo_path.exists():
        logo_bytes = open(logo_path, "rb").read()
        st.markdown(
            """
            <style>
            .robot-logo {
                display: flex;
                justify-content: center;
                margin-top: 0.5rem;
                margin-bottom: 0.5rem;   /* 🔹 Kéo sát tiêu đề hơn */
            }
            video.robot-icon {
                width: 70px;             /* 👈 Nhỏ gọn, chỉ bằng ~1/10 màn hình */
                height: 70px;
                border-radius: 50%;
                object-fit: cover;
                animation: fadeIn 1.5s ease-in-out;
            }
            @keyframes fadeIn {
                from {opacity: 0;}
                to {opacity: 1;}
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # Hiển thị logo động
        st.markdown('<div class="robot-logo">', unsafe_allow_html=True)
        st.video(logo_bytes, start_time=0, loop=True)
        st.markdown("</div>", unsafe_allow_html=True)


    # --- Tiêu đề chính ---
    st.markdown(f"""
        <div style='text-align: center; padding-top: 0.5rem;'>
            <h1 style='font-size: 1.8rem; font-weight: 700; color: white; margin-bottom: 0.4rem;'>
                {title}
            </h1>
            <hr style='width: 80%; margin: 0.2rem auto; border: 0.5px solid #2E2E2E;' />
        </div>
    """, unsafe_allow_html=True)

    # --- Khung chat ---
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    if "chat_histories" not in st.session_state:
        st.session_state.chat_histories = {}
    if st.session_state.current_chat not in st.session_state.chat_histories:
        st.session_state.chat_histories[st.session_state.current_chat] = []

    for role, msg in st.session_state.chat_histories[st.session_state.current_chat]:
        css = "user-message" if role == "user" else "bot-message"
        st.markdown(f'<div class="{css}">{msg}</div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # --- Chat input ---
    # ✨ Dòng hướng dẫn nhập câu hỏi (hiệu ứng fade-in mượt)
    # ✨ Dòng hướng dẫn nhập câu hỏi (fade-in + hiệu ứng ánh sáng)
    st.markdown(
        """
        <style>
        /* 🔹 Hiệu ứng fade-in mượt */
        @keyframes fadeInText {
            0% { opacity: 0; transform: translateY(8px); }
            100% { opacity: 1; transform: translateY(0); }
        }

        /* 🔹 Hiệu ứng ánh sáng lan tỏa (glow pulse) */
        @keyframes glowPulse {
            0% { text-shadow: 0 0 5px #00BFFF, 0 0 10px #00BFFF, 0 0 20px #00BFFF; }
            50% { text-shadow: 0 0 8px #1E90FF, 0 0 16px #1E90FF, 0 0 30px #1E90FF; }
            100% { text-shadow: 0 0 5px #00BFFF, 0 0 10px #00BFFF, 0 0 20px #00BFFF; }
        }

        .prompt-hint {
            text-align: center;
            color: #e0e0e0;                 /* Màu chữ sáng nhẹ */
            font-size: 1.3rem;              /* Cỡ chữ lớn hơn chút */
            font-weight: 600;               /* Đậm vừa phải */
            margin-top: 0.8rem;
            margin-bottom: 1rem;
            animation:
                fadeInText 1.2s ease-in-out,  /* Hiệu ứng hiện mượt */
                glowPulse 2.5s infinite ease-in-out;  /* Hiệu ứng ánh sáng nhịp nhẹ */
        }
        </style>

        <p class="prompt-hint">Please enter your question here</p>
        """,
        unsafe_allow_html=True
    )


    chat_col1, chat_col2 = st.columns([10, 1])
    with chat_col1:
        user_msg = st.chat_input("Nhập câu hỏi hoặc nội dung...")
        st.markdown("""
            <style>
            div[data-baseweb="textarea"] textarea {
                min-height: 80px !important;
                font-size: 1.05rem !important;
                line-height: 1.4 !important;
                padding-top: 0.6rem !important;
            }
            </style>
        """, unsafe_allow_html=True)
    with chat_col2:
        uploaded_files = st.file_uploader("📎", accept_multiple_files=True, label_visibility="collapsed")

    if uploaded_files:
        for file in uploaded_files:
            st.session_state.chat_histories[st.session_state.current_chat].append(
                ("user", f"<div class='file-message'>📄 <b>{file.name}</b></div>")
            )

    # --- Footer chuyên nghiệp ---
    st.markdown(
        "<hr><p style='text-align:center; color:gray; font-size:0.9rem;'>"
        "🤖 The best intelligent assistant for <b>Tom Tran</b> — powered by GPT & RAG"
        "</p>",
        unsafe_allow_html=True
    )

    return user_msg, temperature, top_p, fallback_general, K, MIN_RELEVANCE, debug_mode, show_system, show_rag
