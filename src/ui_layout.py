# src/ui_layout.py
# -*- coding: utf-8 -*-
"""
Giao diện giống ChatGPT thật — full screen, chat box dính đáy
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

    /* 🌑 Toàn bộ nền của ứng dụng */
    body { 
        background-color: #0E1117;   /* Màu nền chính của toàn trang (đen-xám) */
        color: white;                /* Màu chữ mặc định */
    }

    /* 🎛️ Sidebar (cột bên trái) */
    [data-testid="stSidebar"] {
        background-color: #1C1F26;   /* Nền sidebar tối hơn phần chính */
        border-right: 1px solid #2E2E2E; /* Viền ngăn cách giữa sidebar và phần chat */
    }

    /* 🧾 Tiêu đề h1 của trang */
    h1 { 
        text-align: center;          /* Căn giữa tiêu đề */
        font-weight: 700;            /* Chữ đậm */
        color: white;                /* Màu trắng */
    }

    /* 💬 Khung chứa nội dung hội thoại */
    .chat-container {
        display: flex;               /* Sử dụng Flexbox để sắp xếp tin nhắn dọc */
        flex-direction: column;      /* Tin nhắn xếp từ trên xuống dưới */
        justify-content: flex-end;   /* Canh các tin nhắn xuống đáy khung */
        height: 10vh;                /* Chiều cao chiếm 70% màn hình (viewport height) */
        overflow-y: auto;            /* Tự thêm thanh cuộn nếu quá dài */
        padding: 1rem 2rem;          /* Khoảng cách bên trong (trên/dưới 1rem, trái/phải 2rem) */
        background-color: #0E1117;   /* Màu nền giống phần thân để liền khối */
        border-radius: 10px;         /* Bo tròn góc khung chat */
        margin-top: 0.1rem;          /* Giảm khoảng trống phía trên (dưới tiêu đề) */
    }

    /* 🧍 Tin nhắn của người dùng */
    .user-message {
        background-color: #2A2D34;   /* Nền bong bóng người dùng (xám đậm hơn nền chat) */
        padding: 12px 16px;          /* Khoảng cách trong bong bóng */
        border-radius: 12px;         /* Bo góc mềm mại */
        margin: 8px 0;               /* Cách nhau giữa các tin nhắn */
        text-align: right;           /* Căn phải nội dung */
        max-width: 75%;              /* Giới hạn độ rộng tối đa của bong bóng */
        align-self: flex-end;        /* Đặt bong bóng về phía bên phải */
    }

    /* 🤖 Tin nhắn của chatbot */
    .bot-message {
        background-color: #1E1F25;   /* Nền bong bóng của bot (xám đậm hơn người dùng chút) */
        padding: 12px 16px;          /* Khoảng cách trong */
        border-radius: 12px;         /* Bo góc */
        margin: 8px 0;               /* Cách giữa các tin nhắn */
        text-align: left;            /* Căn trái nội dung */
        max-width: 75%;              /* Giới hạn độ rộng */
        align-self: flex-start;      /* Đặt bong bóng về bên trái */
    }

    /* 📎 Tin nhắn kiểu file đính kèm */
    .file-message {
        background-color: #1E252F;   /* Màu riêng cho file */
        border-radius: 8px;          /* Bo tròn nhẹ */
        padding: 10px;               /* Khoảng cách bên trong */
        margin: 6px 0;               /* Cách nhau giữa các file */
        font-size: 0.9em;            /* Chữ nhỏ hơn một chút */
        color: #ccc;                 /* Màu chữ xám nhạt */
    }

    /* 📝 Thanh nhập tin nhắn (input chat) */
    .stChatInputContainer {
        position: fixed;             /* Giữ cố định ở đáy màn hình, không cuộn theo nội dung */
        bottom: 1rem;                /* Cách đáy màn hình 1rem (16px) */
        left: 21rem;                 /* Dời sang phải để không đè lên sidebar */
        right: 2rem;                 /* Cách mép phải 2rem (khoảng trống cân đối) */
        background-color: #0E1117;   /* Nền cùng màu với trang */
    }

    /* ✍️ Ô nhập tin nhắn */
    .stChatInputContainer textarea {
        border-radius: 8px !important;        /* Bo tròn góc input */
        background-color: #1C1F26 !important; /* Màu nền của ô nhập (xám đậm) */
        color: white !important;              /* Màu chữ trắng */
    }

    /* 📱 Tinh chỉnh khi xem trên màn hình nhỏ (responsive) */
    @media (max-width: 900px) {
        .stChatInputContainer {
            left: 1rem;     /* Khi trên mobile, để input chiếm toàn chiều ngang */
            right: 1rem;
        }
    }

    </style>
    """, unsafe_allow_html=True)


    # ========== SIDEBAR ============
    with st.sidebar:
        st.markdown(f"## 💬 {title}")

        if "chat_sessions" not in st.session_state:
            st.session_state.chat_sessions = ["Chat #1"]

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

        # --- toggle hiển thị ---
        show_debug_ui = os.getenv("SHOW_DEBUG_UI", "true").lower() == "true"

        if show_debug_ui:
            st.markdown("### 👁️ Hiển thị các phần")
            show_settings = st.checkbox("⚙️ Cài đặt mô hình", value=True)
            show_system = st.checkbox("📜 System prompt", value=False)
            show_rag = st.checkbox("🧠 RAG diagnostics", value=False)
        else:
            show_settings = False
            show_system = False
            show_rag = False

        # --- cài đặt mô hình ---
        temperature = 0.3
        top_p = 1.0
        fallback_general = True
        K = 4
        MIN_RELEVANCE = 0.3
        debug_mode = False

        if show_debug_ui and show_settings:
            st.markdown("### ⚙️ Cài đặt mô hình")
            temperature = st.slider("Temperature", 0.0, 1.2, 0.3, 0.1)
            top_p = st.slider("Top_p", 0.1, 1.0, 1.0, 0.1)
            fallback_general = st.checkbox("Fallback GPT nếu không có tài liệu phù hợp", value=True)
            K = st.slider("Số đoạn context (k)", 1, 12, 4, 1)
            MIN_RELEVANCE = st.slider("Ngưỡng điểm liên quan tối thiểu (0–1, cao = chặt)", 0.0, 1.0, 0.30, 0.05)
            debug_mode = st.checkbox("🔧 Debug Mode", value=False)

    # ========== MAIN CHAT AREA ============
    # === Tiêu đề chính, có icon robot ===
    st.markdown(f"""
        <div style='text-align: center; padding-top: 0.5rem;'>
            <h1 style='font-size: 1.8rem; font-weight: 700; color: white; margin-bottom: 0.4rem;'>
                🤖{title}
            </h1>
            <hr style='width: 80%; margin: 0.2rem auto; border: 0.5px solid #2E2E2E;' />
        </div>
    """, unsafe_allow_html=True)


    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    if "chat_histories" not in st.session_state:
        st.session_state.chat_histories = {}
    if st.session_state.current_chat not in st.session_state.chat_histories:
        st.session_state.chat_histories[st.session_state.current_chat] = []

    # hiển thị tin nhắn cũ còn giữi lại
    for role, msg in st.session_state.chat_histories[st.session_state.current_chat]:
        css = "user-message" if role == "user" else "bot-message"
        st.markdown(f'<div class="{css}">{msg}</div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # --- chat input & file attach (góc phải, nhỏ) ---
    chat_col1, chat_col2 = st.columns([10, 1])
    with chat_col1:
        user_msg = st.chat_input("Nhập câu hỏi hoặc nội dung...")
    with chat_col2:
        uploaded_files = st.file_uploader("📎", accept_multiple_files=True, label_visibility="collapsed")

    if uploaded_files:
        for file in uploaded_files:
            st.session_state.chat_histories[st.session_state.current_chat].append(
                ("user", f"<div class='file-message'>📄 <b>{file.name}</b></div>")
            )

    return user_msg, temperature, top_p, fallback_general, K, MIN_RELEVANCE, debug_mode, show_system, show_rag
