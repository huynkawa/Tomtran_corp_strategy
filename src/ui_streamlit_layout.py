# src/ui_streamlit_layout.py
# -*- coding: utf-8 -*-
"""
UI ChatGPT-like cho TOMTRAN Chatbot
- Sidebar TRÁI hiển thị lịch sử hội thoại (giống ChatGPT)
- Ô chat NẰM GIỮA header và strap; Enter = gửi (Shift+Enter = xuống dòng)
- Khung chat + ô nhập thu gọn, responsive iPad/mobile
- Có Browse files để upload (PDF/DOCX/TXT)
"""

import base64
import streamlit as st
from pathlib import Path
from docx import Document
from PyPDF2 import PdfReader
from src.chat_saver import save_chat
from src.ui_streamlit_theme import load_theme

ICON_TOP   = "🤖"
ICON_BRAND = "🧭"
USE_FIXED_CHAT_INPUT = False
LOGO_RELATIVE_PATH = Path("assets") / "Dancing Chatbot.mp4"

PROMPT_TEXT = "Chúng ta nên bắt đầu từ đâu - Hỏi bất cứ điều gì"
PROMPT_COLOR = "#FFFFFF"
PROMPT_FONT_REM = 1.20

ALLOW_UPLOAD = True
UPLOAD_TYPES = ["pdf", "docx", "txt"]


def read_text_from_file(uploaded_file):
    try:
        name = uploaded_file.name.lower()
        if name.endswith(".txt"):
            return uploaded_file.read().decode("utf-8", errors="ignore")
        if name.endswith(".pdf"):
            rdr, text = PdfReader(uploaded_file), ""
            for p in rdr.pages[:3]:
                text += (p.extract_text() or "") + "\n"
            return text or "[Không đọc được nội dung PDF]"
        if name.endswith(".docx"):
            doc = Document(uploaded_file)
            text = "\n".join(p.text for p in doc.paragraphs[:30])
            return text or "[Không đọc được nội dung DOCX]"
        return "[Định dạng không hỗ trợ preview]"
    except Exception as e:
        return f"[Lỗi đọc file: {e}]"


def _chat_label(chat_name: str) -> str:
    hist = st.session_state.chat_histories.get(chat_name, [])
    for role, msg in hist:
        if role == "user":
            s = (msg or "").strip().replace("\n", " ")
            return s[:40] + ("…" if len(s) > 40 else "")
    return chat_name


def render_ui(title: str = "TOMTRANCHATBOT"):
    theme = load_theme()
    st.set_page_config(page_title=title, layout="wide")

    ss = st.session_state
    ss.setdefault("chat_sessions", ["Đoạn Chat 1"])
    ss.setdefault("chat_histories", {})
    ss.setdefault("current_chat", ss["chat_sessions"][-1])
    if ss["current_chat"] not in ss["chat_sessions"]:
        ss["current_chat"] = ss["chat_sessions"][-1]
    current_chat = ss["current_chat"]
    if current_chat not in ss["chat_histories"]:
        ss["chat_histories"][current_chat] = []

    ss.setdefault("center_text", "")
    ss.setdefault("clear_center_input", False)
    ss.setdefault("last_handled_msg", "")

    if ss.get("clear_center_input"):
        ss["center_text"] = ""
        ss["clear_center_input"] = False

    is_empty = (len(ss.chat_histories[current_chat]) == 0)
    chat_height_vh = theme.CHAT_HEIGHT_VH_EMPTY if is_empty else theme.CHAT_HEIGHT_VH_NONEMPTY

    # =================== CSS ===================
    st.markdown(f"""
    <style>
    :root {{
        --bg:#0E1117; --card:#1C1F26; --line:#2E2E2E; --text:#fff; --muted:#bfbfbf;
        --chat-width-desktop: {theme.CHAT_WIDTH_DESKTOP_PX}px;
        --chat-width-tablet:  {theme.CHAT_WIDTH_TABLET_PX}px;
        --chat-width-mobile:  {theme.CHAT_WIDTH_MOBILE_PCT}%;
    }}
    body {{ background-color: var(--bg); color: var(--text); }}

    /* Sidebar trái */
    [data-testid="stSidebar"]{{
        width: {theme.LEFT_SIDEBAR_WIDTH_PX}px;
        min-width: {theme.LEFT_SIDEBAR_WIDTH_PX}px;
        background-color: var(--card);
        border-right: 1px solid var(--line);
    }}
    [data-testid="stSidebar"] .stButton > button {{
        justify-content: flex-start !important;
        text-align: left !important;
    }}

    /* Header (logo + tiêu đề) */
    .tt-header{{
        display:flex; align-items:center; justify-content:center; gap:.6rem; padding:.6rem 0 .2rem 0;
        margin-top:{theme.LOGO_OFFSET_REM}rem;
    }}
    .tt-title{{ font-weight:800; font-size:{theme.TITLE_FONT_SIZE_REM}rem; letter-spacing:.3px; }}
    .tt-logo{{ margin-top:{theme.LOGO_OFFSET_REM}rem; }}

    .tt-logo video{{
        width:{theme.LOGO_SIZE_PX}px; height:{theme.LOGO_SIZE_PX}px;
        border-radius:{getattr(theme,"LOGO_BORDER_RADIUS_PX",50)}px;
        object-fit:{getattr(theme,"LOGO_OBJECT_FIT","cover")};
        {"box-shadow:0 0 10px rgba(0,191,255,.5);" if getattr(theme,"LOGO_SHADOW", True) else "box-shadow:none;"}
    }}
    .tt-logo img{{
        width:{theme.LOGO_SIZE_PX}px; height:{theme.LOGO_SIZE_PX}px;
        border-radius:{getattr(theme,"LOGO_BORDER_RADIUS_PX",50)}px;
        object-fit:{getattr(theme,"LOGO_OBJECT_FIT","cover")};
        {"box-shadow:0 0 10px rgba(0,191,255,.5);" if getattr(theme,"LOGO_SHADOW", True) else "box-shadow:none;"}
    }}

    /* Đường kẻ kiểu cũ */
    .tt-hr{{
        width:{getattr(theme,"HEADER_DIVIDER_WIDTH_PCT",82)}%;
        margin:{getattr(theme,"HEADER_DIVIDER_TOP_REM",0.6)}rem auto {getattr(theme,"HEADER_DIVIDER_BOTTOM_REM",0.35)}rem auto;
        border:0;
        border-top:{getattr(theme,"HEADER_DIVIDER_THICK_PX",1)}px {getattr(theme,"HEADER_DIVIDER_STYLE","solid")} {getattr(theme,"HEADER_DIVIDER_COLOR","#2E2E2E")};
        border-radius:2px;
    }}

    /* Banner chữ + 2 vạch hai bên */
    .tt-banner{{
        display:flex; align-items:center; justify-content:center; gap:.8rem;
        width:{getattr(theme,"HEADER_DIVIDER_WIDTH_PCT",82)}%;
        margin:{getattr(theme,"HEADER_DIVIDER_TOP_REM",0.6)}rem auto {getattr(theme,"HEADER_DIVIDER_BOTTOM_REM",0.35)}rem auto;
        color:{getattr(theme,"HEADER_DIVIDER_TEXT_COLOR","#bfbfbf")};
        font-size:{getattr(theme,"HEADER_DIVIDER_TEXT_SIZE_REM",0.95)}rem;
        font-weight:{getattr(theme,"HEADER_DIVIDER_TEXT_WEIGHT",600)};
        text-align:center;
    }}
    .tt-banner::before, .tt-banner::after{{
        content:""; flex:1;
        border-top:{getattr(theme,"HEADER_DIVIDER_THICK_PX",1)}px {getattr(theme,"HEADER_DIVIDER_STYLE","solid")} {getattr(theme,"HEADER_DIVIDER_COLOR","#2E2E2E")};
        opacity:.9;
    }}

    /* Bọc ngoài để giới hạn bề rộng */
    .chat-wrapper{{
        max-width: var(--chat-width-desktop);
        margin: {theme.HEADER_TO_CHAT_GAP_REM}rem auto 0 auto;
        padding:0;
    }}
    .chat-container{{
        display:flex; flex-direction:column; justify-content:flex-end;
        height:{chat_height_vh}vh;
        overflow-y:auto; background:var(--bg);
        border-radius:12px;
        padding:{theme.CHAT_PAD_V_REM}rem {theme.CHAT_PAD_H_REM}rem;
    }}
    .user-message,.bot-message{{ padding:12px 16px; border-radius:12px; margin:8px 0; max-width:75%; }}
    .user-message{{ background:#2A2D34; align-self:flex-end; text-align:right; }}
    .bot-message{{ background:#1E1F25; align-self:flex-start; text-align:left; }}

    .prompt-hint{{ text-align:center; color:{PROMPT_COLOR}; font-weight:700; font-size:{PROMPT_FONT_REM}rem; margin:.45rem 0 .55rem 0; }}

    /* Ô nhập (nếu dùng st.chat_input) */
    .stChatInputContainer{{
        position:relative; margin-top:{theme.INPUT_MARGIN_TOP_REM}rem !important;
        max-width: var(--chat-width-desktop);
        margin-left:auto !important; margin-right:auto !important; z-index:2;
    }}
    div[data-baseweb="textarea"] textarea{{
        min-height:84px !important; font-size:1.06rem !important; line-height:1.45 !important;
        background:var(--card) !important; color:#fff !important; border-radius:16px !important;
        padding-top:.9rem !important; box-shadow:0 0 10px rgba(0,0,0,.35) !important;
    }}

    /* Responsive */
    @media (max-width:1200px){{
        .chat-wrapper{{ max-width: var(--chat-width-tablet); }}
        .stChatInputContainer{{ max-width: var(--chat-width-tablet); }}
    }}
    @media (max-width:900px){{
        .chat-wrapper{{ max-width: var(--chat-width-mobile); margin-left:auto; margin-right:auto; }}
        .stChatInputContainer{{ max-width: var(--chat-width-mobile); }}
        .chat-container{{ height:{max(50, chat_height_vh-7)}vh !important; padding:.6rem 1rem !important; }}
        .tt-title{{ font-size:{max(1.3, theme.TITLE_FONT_SIZE_REM-0.4)}rem; }}
    }}

    /* Thu gọn nút Gửi trong form giữa */
    .chat-wrapper form .stButton > button{{
        padding:.55rem 1rem;
        border-radius:10px;
        font-weight:600;
    }}
    </style>
    """, unsafe_allow_html=True)

    # ======= SIDEBAR =======
    with st.sidebar:
        st.markdown(f"### {ICON_TOP} {title}")
        if st.button("➕ Tạo đoạn chat mới", use_container_width=True):
            from datetime import datetime
            safe_ts = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
            new_name = f"Chat_{safe_ts}"
            ss.chat_sessions.append(new_name)
            ss.chat_histories[new_name] = []
            ss.current_chat = new_name
            save_chat(new_name, [])
            st.rerun()

        st.divider()
        st.caption("Đoạn chat")
        active = ss.get("current_chat", ss.chat_sessions[-1])
        for idx, name in enumerate(ss.chat_sessions):
            label = _chat_label(name)
            is_active = (name == active)
            btn_label = ("• " if is_active else "") + label
            if st.button(btn_label, key=f"left_{idx}", use_container_width=True):
                ss.current_chat = name
                st.rerun()

    # ======= HEADER (logo + tiêu đề) =======
    logo_html = ""
    logo_path_str = (
        getattr(theme, "RESOLVED_LOGO_PATH", None)
        or getattr(theme, "resolved_logo_path", None)
        or getattr(theme, "LOGO_PATH", "")
        or ""
    )

    def _is_http_url(s: str) -> bool:
        return s.startswith("http://") or s.startswith("https://")

    def _abs_path(p: str) -> Path:
        pth = Path(p)
        if pth.is_absolute():
            return pth
        base_dir = Path(__file__).resolve().parents[1]
        return (base_dir / p).resolve()

    try:
        if logo_path_str:
            if _is_http_url(logo_path_str):
                if theme.LOGO_IS_VIDEO:
                    logo_html = (
                        f'<div class="tt-logo"><video autoplay loop muted playsinline '
                        f'width="{theme.LOGO_SIZE_PX}" height="{theme.LOGO_SIZE_PX}">'
                        f'<source src="{logo_path_str}" type="video/mp4"></video></div>'
                    )
                else:
                    logo_html = (
                        f'<div class="tt-logo"><img src="{logo_path_str}" alt="logo" '
                        f'style="width:{theme.LOGO_SIZE_PX}px;height:{theme.LOGO_SIZE_PX}px;'
                        f'border-radius:{getattr(theme,"LOGO_BORDER_RADIUS_PX",50)}px;'
                        f'object-fit:{getattr(theme,"LOGO_OBJECT_FIT","cover")};"/></div>'
                    )
            else:
                lp = _abs_path(logo_path_str)
                if lp.exists():
                    b64 = base64.b64encode(lp.read_bytes()).decode("utf-8")
                    if theme.LOGO_IS_VIDEO and lp.suffix.lower() in {".mp4", ".webm"}:
                        mime = "video/mp4" if lp.suffix.lower() == ".mp4" else "video/webm"
                        logo_html = (
                            f'<div class="tt-logo"><video autoplay loop muted playsinline '
                            f'width="{theme.LOGO_SIZE_PX}" height="{theme.LOGO_SIZE_PX}">'
                            f'<source src="data:{mime};base64,{b64}" type="{mime}"></video></div>'
                        )
                    else:
                        ext = lp.suffix.lower()
                        mime = "image/png" if ext == ".png" else ("image/jpeg" if ext in {".jpg",".jpeg"} else "image/*")
                        logo_html = (
                            f'<div class="tt-logo"><img src="data:{mime};base64,{b64}" alt="logo" '
                            f'style="width:{theme.LOGO_SIZE_PX}px;height:{theme.LOGO_SIZE_PX}px;'
                            f'border-radius:{getattr(theme,"LOGO_BORDER_RADIUS_PX",50)}px;'
                            f'object-fit:{getattr(theme,"LOGO_OBJECT_FIT","cover")};"/></div>'
                        )
    except Exception:
        logo_html = ""

    if not logo_html:
        fallback = Path(__file__).resolve().parents[1] / "assets" / "logo.png"
        if fallback.exists():
            b64 = base64.b64encode(fallback.read_bytes()).decode("utf-8")
            logo_html = (
                f'<div class="tt-logo"><img src="data:image/png;base64,{b64}" alt="logo" '
                f'style="width:{theme.LOGO_SIZE_PX}px;height:{theme.LOGO_SIZE_PX}px;'
                f'border-radius:{getattr(theme,"LOGO_BORDER_RADIUS_PX",50)}px;'
                f'object-fit:{getattr(theme,"LOGO_OBJECT_FIT","cover")};"/></div>'
            )
        else:
            logo_html = ""

    title_html = f'<div class="tt-title">{title}</div>' if theme.SHOW_TITLE else ""

    st.markdown(
        f"""
        <div class="tt-header">
            {logo_html}
            {title_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ======= Banner chữ / đường kẻ ngay dưới header =======
    mode = getattr(theme, "HEADER_DIVIDER_MODE", "line")
    text = getattr(theme, "HEADER_DIVIDER_TEXT", "").strip()
    banner_html = f'<div class="tt-banner">{text}</div>'
    line_html   = '<hr class="tt-hr"/>'
    st.markdown(
        banner_html if (mode == "text" and text) else (line_html if getattr(theme, "SHOW_HEADER_DIVIDER", True) else ""),
        unsafe_allow_html=True,
    )

    # ======= LỊCH SỬ CHAT =======
    st.markdown('<div class="chat-wrapper"><div class="chat-container">', unsafe_allow_html=True)
    for role, msg in ss.chat_histories[current_chat]:
        css = "user-message" if role == "user" else "bot-message"
        st.markdown(f'<div class="{css}">{msg}</div>', unsafe_allow_html=True)
    st.markdown("</div></div>", unsafe_allow_html=True)

    # ======= Ô NHẬP =======
    user_msg = ""
    uploaded_files = []
    if not USE_FIXED_CHAT_INPUT:
        st.markdown('<div class="chat-wrapper">', unsafe_allow_html=True)
        with st.form("center_input"):
            if ALLOW_UPLOAD:
                uploaded_files = st.file_uploader(
                    "Tài liệu đính kèm",
                    type=UPLOAD_TYPES,
                    accept_multiple_files=True,
                    label_visibility="collapsed"
                )
                if uploaded_files:
                    ss["last_uploaded_files"] = uploaded_files

            user_msg = st.text_area(
                "Tin nhắn",
                key="center_text",
                placeholder="Hỏi bất kỳ điều gì …",
                height=88,
                label_visibility="collapsed"
            )

            c_spacer, c_btn = st.columns([1, 0.18])
            with c_btn:
                sent = st.form_submit_button("Gửi", use_container_width=True)

            st.markdown("""
            <script>
            const root = window.parent.document;
            const ta = root.querySelector('textarea[aria-label="Tin nhắn"]');
            const btns = Array.from(root.querySelectorAll('button'));
            const btn = btns.find(b => b.innerText.trim() === 'Gửi');
            if (ta && btn){
              ta.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && !e.shiftKey){
                  e.preventDefault();
                  btn.click();
                }
              });
            }
            </script>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        if ALLOW_UPLOAD:
            st.file_uploader("Tài liệu đính kèm", type=UPLOAD_TYPES, accept_multiple_files=True)
        user_msg = st.chat_input("Hỏi bất kỳ điều gì …")

    # ======= STRAP / FOOTER =======
    if theme.SHOW_STRAP:
        st.markdown(
            f'<div style="text-align:center; color:#cfcfcf; font-size:.95rem; '
            f'margin:{theme.FOOTER_STRAP_TOP_MARG_REM}rem 0 .2rem 0;">'
            f'{theme.STRAP_ICON} {theme.STRAP_TEXT}</div>',
            unsafe_allow_html=True,
        )

    if theme.SHOW_FOOTER:
        st.markdown(
            f"""
            <div style="text-align:center; color:#9a9a9a; font-size:.92rem; line-height:1.55;
                        margin-top:{theme.FOOTER_BRAND_TOP_MARG_REM}rem;">
                <span style="font-weight:600; color:#ccc;">{theme.FOOTER_TITLE}</span> —
                <span style="color:#aaa;">{theme.FOOTER_TAGLINE}</span><br/>
                <span style="font-size:.85rem; color:#666;">{theme.FOOTER_COPYRIGHT}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ======= RETURN =======
    temperature = 0.3
    top_p = 1.0
    fallback_general = True
    K = 4
    MIN_RELEVANCE = 0.30
    debug_mode = False
    show_system = False
    show_rag = False

    return (
        (user_msg or "").strip(),
        float(temperature),
        float(top_p),
        bool(fallback_general),
        int(K),
        float(MIN_RELEVANCE),
        bool(debug_mode),
        bool(show_system),
        bool(show_rag),
    )
