# src/ui_streamlit_layout.py
# -*- coding: utf-8 -*-
"""
UI Chat kiểu ChatGPT — full screen, sidebar trái, khung chat có viền, input dính đáy
"""

import os
import streamlit as st
from io import BytesIO
from docx import Document
from PyPDF2 import PdfReader
from src.chat_saver import save_chat

# --- Footer từ YAML (robust, không bị dư </p>) ---
import yaml
from pathlib import Path

def _load_theme_yaml(path="configs/ui_streamlit_theme.yaml"):
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}

_cfg = _load_theme_yaml()
_ft  = (_cfg.get("FOOTER") or {})

if _ft.get("ENABLED", True):
    text_html  = (_ft.get("TEXT_HTML") or "").strip()
    align      = _ft.get("ALIGN", "center")
    color      = _ft.get("COLOR", "#6B7280")
    size_rem   = _ft.get("SIZE_REM", 0.9)
    show_hr_t  = _ft.get("SHOW_HR_TOP", True)
    show_hr_b  = _ft.get("SHOW_HR_BOTTOM", False)
    hr_color   = _ft.get("HR_COLOR", "#E5E7EB")
    hr_thick   = int(_ft.get("HR_THICK_PX", 1))
    mt         = _ft.get("MARGIN_TOP_REM", 0.6)
    mb         = _ft.get("MARGIN_BOTTOM_REM", 0.0)

    hr_top = f"<hr style='border:0;border-top:{hr_thick}px solid {hr_color};margin:.4rem 0;'/>" if show_hr_t else ""
    hr_bot = f"<hr style='border:0;border-top:{hr_thick}px solid {hr_color};margin:.4rem 0;'/>" if show_hr_b else ""

    if text_html.startswith("<"):
        body = (
            f'<div style="text-align:{align}; color:{color}; '
            f'font-size:{size_rem}rem; margin:{mt}rem 0 {mb}rem;">{text_html}</div>'
        )
    else:
        body = (
            f'<p style="text-align:{align}; color:{color}; '
            f'font-size:{size_rem}rem; margin:{mt}rem 0 {mb}rem;">{text_html}</p>'
        )

    st.markdown(f"{hr_top}{body}{hr_bot}", unsafe_allow_html=True)


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


def render_ui(title="TomTran Strategy AI"):
    # 1) Page config phải đặt TRƯỚC mọi st.* khác
    st.set_page_config(page_title=title, page_icon="💬", layout="wide")

    from src.ui_streamlit_theme_old import load_theme
    # ĐỌC THEME TỪ YAML + KHAI BÁO BIẾN DÙNG CHO CSS
    theme = load_theme()
    input_text = getattr(theme, "INPUT_TEXT", "#111827")
    input_txt = input_text
    # Kích thước khung chat
    main_max   = getattr(theme, "CHAT_WIDTH_DESKTOP_PX", 820)
    tablet_max = getattr(theme, "CHAT_WIDTH_TABLET_PX", 680)
    mobile_pct = getattr(theme, "CHAT_WIDTH_MOBILE_PCT", 96)

    # Ô nhập (màu/viền/chiều cao)
    input_bg   = getattr(theme, "INPUT_BG", "#E5F1FF")
    input_text = getattr(theme, "INPUT_TEXT", "#111827")        # <-- cái bạn đang thiếu
    input_brd  = getattr(theme, "INPUT_BORDER_DARK", "#BFD9FF")
    input_h    = getattr(theme, "INPUT_MIN_HEIGHT_PX", 100)

    # Hint/tiêu đề
    prompt_col = getattr(theme, "PROMPT_TEXT", "#111827")       # <-- cái bạn đang thiếu
    prompt_sz  = getattr(theme, "PROMPT_FONT_SIZE_REM", 1.0)    # <-- cái bạn đang thiếu
    title_sz   = getattr(theme, "TITLE_FONT_SIZE_REM", 1.8)     # <-- cái bạn đang thiếu

    # CSS “chat input”
    # ==== CSS for chat input (always visible) ====
    st.markdown(
        f"""
    <style>
    :root{{
        --main-max: {main_max}px;
        --blue-100: {input_bg};
        --blue-300: {input_brd};
    }}

    /* Chat history container */
    .chat-container{{
        width: 80%;
        max-width: var(--main-max);
        margin: 0 auto 8px auto !important;
        background: transparent !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0;
        box-shadow: none !important;
        min-height: 0 !important;
    }}

    .chat-container:has(.user-message),
    .chat-container:has(.bot-message){{
        background: #F3F4F6 !important;
        border: 1.5px solid #9CA3AF !important;
        padding: 10px 14px;
        min-height: 0 !important;
    }}

    /* Hint just above the input */
    .prompt-hint{{
        width:75% !important;
        max-width: calc(var(--main-max) * 0.75) !important;
        margin: 4px auto 4px auto !important;
        text-align: left !important;
        color: {prompt_col};
        font-size:{prompt_sz}rem;
        font-weight:500;
    }}

    /* Chat input container – support both selectors */
    .stChatInputContainer,
    [data-testid="stChatInput"]{{
        position: static !important;   /* avoid sticky push-down */
        width:75% !important;
        max-width: calc(var(--main-max) * 0.75) !important;
        margin: 6px auto 10px auto !important;
        box-sizing: border-box !important;
        opacity:1 !important;
        visibility:visible !important;
        pointer-events:auto !important;
    }}

    /* Outer wrapper of textarea */
    .stChatInputContainer [data-baseweb="textarea"],
    [data-testid="stChatInput"] [data-baseweb="textarea"]{{
        width: 100% !important;
        min-height: {input_h}px !important;
        background: var(--blue-100) !important;
        border: 1.5px solid var(--blue-300) !important;
        border-radius: 12px !important;
        box-shadow: none !important;
    }}

    /* Actual textarea */
    .stChatInputContainer [data-baseweb="textarea"] textarea,
    [data-testid="stChatInput"] [data-baseweb="textarea"] textarea{{
        min-height: {input_h}px !important;
        color: {input_text} !important;
        background: transparent !important;
        border: 0 !important;
        outline: none !important;
        box-shadow: none !important;
        border-radius: 12px !important;
        padding-top: .8rem !important;
        text-align: left !important;
    }}

    .stChatInputContainer [data-baseweb="textarea"] textarea::placeholder,
    [data-testid="stChatInput"] [data-baseweb="textarea"] textarea::placeholder{{
        color: #6B728099 !important;
    }}

    /* Responsive from YAML */
    @media (max-width:1200px){{
        .chat-container{{ max-width: {tablet_max}px; }}
    }}
    @media (max-width:900px){{
        .chat-container{{ width:{mobile_pct}%; max-width:none; }}
        .stChatInputContainer,
        [data-testid="stChatInput"]{{ width:{mobile_pct}% !important; max-width:none !important; }}
    }}
    </style>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("""
    <style id="chat-input-final-override">
    /* BẮT BUỘC: không dính đáy */
    .stChatInputContainer,
    [data-testid="stChatInput"]{
        position: static !important;
        bottom: auto !important;
        top: auto !important;
        margin: 6px auto 10px auto !important;   /* sát hint */
        opacity: 1 !important;
        visibility: visible !important;
        pointer-events: auto !important;
    }

    /* Không để khung lịch sử chiếm chỗ khi rỗng */
    .chat-container{
        margin-bottom: 0 !important;
        min-height: 0 !important;
        padding-bottom: 0 !important;
    }

    /* Hint mỏng, sát input */
    .prompt-hint{ margin: 4px auto 4px auto !important; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <style id="final-y-compact">
    /* (1) Giảm padding dọc mặc định của trang */
    .main .block-container{
        padding-top: 0.4rem !important;
        padding-bottom: 0.6rem !important;
    }

    /* (2) Tiêu đề: đừng để khoảng cách lớn */
    h1{
        margin-bottom: 6px !important;
    }
    h1 + hr{ display:none !important; }  /* nếu còn <hr> dưới H1 */

    /* (3) Lịch sử chat: khi rỗng không chiếm chỗ, khi có thì padding nhỏ */
    .chat-container{
        min-height: 0 !important;
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
    }
    .chat-container:has(.user-message),
    .chat-container:has(.bot-message){
        padding: 10px 14px !important;   /* giảm so với mặc định */
        margin-bottom: 6px !important;
    }

    /* (4) Hint: sát input */
    .prompt-hint{
        margin: 1px auto 1px auto !important;
    }

    /* (5) Ô chat: KHÔNG sticky, kéo sát lên trên */
    .stChatInputContainer,
    [data-testid="stChatInput"]{
        position: static !important;
        bottom: auto !important;
        margin: 2px auto 8px auto !important;   /* chỉnh con số này nếu muốn sát hơn/xa hơn */
        opacity: 1 !important; visibility: visible !important; pointer-events: auto !important;
    }

    /* (6) Uploader: đừng đẩy mọi thứ ra xa */
    .element-container:has([data-testid="stFileUploader"]){
        margin-bottom: 2px !important;
    }
    </style>
    """, unsafe_allow_html=True)


    # === FINAL OVERRIDES: kéo ô chat lên, đổi màu viền & ẩn drag-drop ===
    input_mt = getattr(theme, "INPUT_MARGIN_TOP_REM", -12)
    input_bg = getattr(theme, "INPUT_BG", "#F9FAFB")
    input_tx = getattr(theme, "INPUT_TEXT", "#111827")
    input_bd = getattr(theme, "INPUT_BORDER_DARK", "#9CA3AF")

    st.markdown(f"""
    <style id="chatgpt-like-overrides">
    /* 1) Kéo ô chat lên gần giữa trang */
    .stChatInputContainer,
    [data-testid="stChatInput"]{{
    position: static !important;
    margin-top: {input_mt}rem !important;   /* ĐẨY LÊN */
    margin-bottom: 0.8rem !important;
    }}

    /* 2) Đổi màu viền đỏ → xám, nền xám nhạt, chặn trạng thái 'invalid' đỏ */
    .stChatInputContainer [data-baseweb="textarea"],
    [data-testid="stChatInput"] [data-baseweb="textarea"],
    .stChatInputContainer [data-baseweb="textarea"]:hover,
    [data-testid="stChatInput"] [data-baseweb="textarea"]:hover,
    .stChatInputContainer [data-baseweb="textarea"]:focus-within,
    [data-testid="stChatInput"] [data-baseweb="textarea"]:focus-within {{
    background: {input_bg} !important;
    border: 1.5px solid {input_bd} !important;
    border-radius: 999px !important;               /* hình viên thuốc giống ChatGPT */
    box-shadow: none !important;
    }}

    .stChatInputContainer [data-baseweb="textarea"] textarea,
    [data-testid="stChatInput"] [data-baseweb="textarea"] textarea{{
    color: {input_tx} !important;
    background: transparent !important;
    border: 0 !important;
    outline: none !important;
    box-shadow: none !important;
    text-align: left !important;
    padding: .70rem 1.0rem !important;
    min-height: 44px !important;
    }}

    /* 3) Ẩn vùng drag & drop + dòng chữ 'Drag and drop files here' + 'Limit ...' */
    [data-testid="stFileUploaderDropzone"],
    [data-testid="stFileUploadDropzone"],
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneDescription"],
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDetails"] {{
    display: none !important;   /* chỉ giữ lại NÚT Browse files */
    }}

    /* 4) Căn nút Browse files sang phải, gọn như ChatGPT */
    .element-container:has([data-testid="stFileUploader"]) {{
    width: 75% !important;
    max-width: calc(var(--main-max, 820px) * 0.75) !important;
    margin: .25rem auto .5rem auto !important;
    display: flex; justify-content: flex-end;
    }}
    [data-testid="stFileUploader"] button {{
    border-radius: 10px !important;
    }}

    /* 5) Thu nhỏ khoảng trắng tổng thể để input nằm gần trung tâm */
    .main .block-container {{
    padding-top: .4rem !important;
    padding-bottom: .6rem !important;
    }}
    </style>
    """, unsafe_allow_html=True)

    # ========== SIDEBAR ==========
    with st.sidebar:
        st.markdown(f"## 💬 {title}")

        if "chat_sessions" not in st.session_state:
            st.session_state.chat_sessions = ["Đoạn Chat 1"]
        if "chat_histories" not in st.session_state:
            st.session_state.chat_histories = {"Đoạn Chat 1": []}
        if "current_chat" not in st.session_state:
            st.session_state.current_chat = "Đoạn Chat 1"

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
            show_system   = st.checkbox("📜 System prompt", value=False)
            show_rag      = st.checkbox("🧠 RAG diagnostics", value=False)
        else:
            show_settings = show_system = show_rag = False

        # Thông số mặc định
        temperature = 0.3
        top_p = 1.0
        fallback_general = True
        K = 4
        MIN_RELEVANCE = 0.3
        debug_mode = False

    # ========== MAIN CHAT AREA ==========
    # Logo động (optional)
    from pathlib import Path
    logo_path = Path("assets/robot_logo.mp4")
    if logo_path.exists():
        with open(logo_path, "rb") as f:
            logo_bytes = f.read()
        st.markdown('<div style="display:flex;justify-content:center;margin:.5rem 0 .5rem 0">', unsafe_allow_html=True)
        st.video(logo_bytes, start_time=0, loop=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Tiêu đề

    st.markdown(f"""
    <div style='text-align:center; padding-top:.3rem;'>
        <h1 style='font-size:{title_sz}rem; font-weight:700; color:#111827; margin-bottom:.4rem;'>{title}</h1>
    </div>
    """, unsafe_allow_html=True)


    # Lịch sử chat
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    if st.session_state.current_chat not in st.session_state.chat_histories:
        st.session_state.chat_histories[st.session_state.current_chat] = []
    for role, msg in st.session_state.chat_histories[st.session_state.current_chat]:
        css = "user-message" if role == "user" else "bot-message"
        st.markdown(f'<div class="{css}">{msg}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Hint căn trái
    st.markdown('<p class="prompt-hint">Enter your question here</p>', unsafe_allow_html=True)



    # Input + Uploader (Browse files ở cột phải)
    # Ô chat đặt độc lập để CSS luôn áp dụng chuẩn
    user_msg = st.chat_input("")

    # Uploader để riêng bên dưới (nếu cần đặt cạnh, ta sẽ làm layout khác sau)
    uploaded_files = st.file_uploader("📎", accept_multiple_files=True, label_visibility="collapsed")


    if uploaded_files:
        for file in uploaded_files:
            st.session_state.chat_histories[st.session_state.current_chat].append(
                ("user", f"<div class='file-message'>📄 <b>{file.name}</b></div>")
            )

    # Footer từ YAML (hot reload)
    _cfg = _load_theme_yaml()
    _ft  = (_cfg.get("FOOTER") or {})

    return user_msg, temperature, top_p, fallback_general, K, MIN_RELEVANCE, debug_mode, show_system, show_rag
