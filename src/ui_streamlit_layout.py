# src/ui_streamlit_layout.py
import streamlit as st
from src.ui_streamlit_theme import get_cfg  # cùng thư mục src, import theo PYTHONPATH của bạn

# --- inject CSS từ cfg (đọc từ ui_streamlit_theme.yaml) ---
def inject_css_from_cfg(cfg: dict):
    co, tp, ly, ca, fx, obj = (
        cfg["COLORS"], cfg["TYPOGRAPHY"], cfg["LAYOUT"], cfg["CENTER_AXIS"], cfg["EFFECTS"], cfg["OBJECTS"]
    )
    sb, ch, ci = obj["SIDEBAR"], obj["CHAT_PANEL"], obj["CHAT_INPUT"]
    div = ly["DIVIDER"]

    center_x = f"{ca['X_PCT']}vw"
    center_y = f"{ca['Y_PCT']}vh"

    st.markdown(f"""
    <style>
      :root {{
        --bg:{co['BG']}; --text:{co['TEXT']}; --card:{co['CARD']}; --line:{co['LINE']};
        --base:{tp['BASE_SIZE_PX']}px; --title:{tp['TITLE_SIZE_PX']}px; --font:{tp['FONT_FAMILY']};
        --main-max:{ly['MAIN_MAX_WIDTH_PCT']}vw; --gutter:{ly['GUTTER_PX']}px;
        --sidebar-w:{ly['SIDEBAR_WIDTH_PCT']}%; --chat-w:{ly['CHAT_WIDTH_PCT']}%;
        --divider-show:{'block' if div['SHOW'] else 'none'};
        --divider-color:{div['COLOR']}; --divider-style:{div['STYLE']}; --divider-width:{div['WIDTH_PX']}px; --divider-ox:{div['OFFSET_X_PX']}px;
        --center-x:{center_x}; --center-y:{center_y};
        --sb-ox:{sb['OFFSET_X']}; --sb-oy:{sb['OFFSET_Y']}; --sb-bg:{sb['BG']}; --sb-text:{sb['TEXT']};
        --sb-bc:{sb['BORDER_COLOR']}; --sb-bw:{sb['BORDER_PX']}px; --sb-r:{sb['RADIUS_PX']}px; --sb-pad:{sb['PADDING_PX']}px;
        --ch-ox:{ch['OFFSET_X']}; --ch-oy:{ch['OFFSET_Y']}; --ch-bg:{ch['BG']}; --ch-text:{ch['TEXT']};
        --ch-bc:{ch['BORDER_COLOR']}; --ch-bw:{ch['BORDER_PX']}px; --ch-r:{ch['RADIUS_PX']}px; --ch-pad:{ch['PADDING_PX']}px;
        --chat-h-d:{ch['CHAT_HEIGHT_VH_DESKTOP']}vh; --chat-h-m:{ch['CHAT_HEIGHT_VH_MOBILE']}vh;
        --ci-w:{ci['WIDTH_PCT']}%; --ci-ox:{ci['OFFSET_X']}; --ci-oy:{ci['OFFSET_Y']};
        --ci-bg:{ci['BG']}; --ci-bc:{ci['BORDER_COLOR']}; --ci-bw:{ci['BORDER_PX']}px; --ci-r:{ci['RADIUS_PX']}px;
      }}

      body,.stApp {{ background: var(--bg)!important; color: var(--text)!important; font-family: var(--font)!important; font-size: var(--base); }}
      section.main > div.block-container {{ max-width: var(--main-max)!important; padding-left: var(--gutter)!important; padding-right: var(--gutter)!important; }}
      section.main > div.block-container > div:first-child {{ display:grid; grid-template-columns:minmax(0,var(--sidebar-w)) minmax(0,var(--chat-w)); gap:0; position:relative; }}
      section.main > div.block-container > div:first-child::before {{
        content:""; display: var(--divider-show); position:absolute; top:0; bottom:0;
        left: calc(var(--sidebar-w) + var(--divider-ox)); width: var(--divider-width);
        background: var(--divider-color); border-left: var(--divider-width) var(--divider-style) var(--divider-color); z-index:9;
      }}
      [data-testid="stSidebar"] > div {{
        background: var(--sb-bg)!important; color: var(--sb-text)!important;
        border: var(--sb-bw) solid var(--sb-bc); border-radius: var(--sb-r); box-shadow: none; padding: var(--sb-pad);
        transform: translate(calc(var(--center-x) + var(--sb-ox) - 50%), calc(var(--center-y) + var(--sb-oy) - 50%));
        transition: transform 180ms ease;
      }}
      section.main > div.block-container > div:first-child > div:last-child {{
        background: var(--ch-bg); color: var(--ch-text);
        border: var(--ch-bw) solid var(--ch-bc); border-radius: var(--ch-r); padding: var(--ch-pad);
        min-height: var(--chat-h-d);
        transform: translate(calc(var(--center-x) + var(--ch-ox) - 50%), calc(var(--center-y) + var(--ch-oy) - 50%));
        transition: transform 180ms ease;
      }}
      [data-testid="stChatInput"] {{ width: var(--ci-w)!important; margin:12px auto; transform: translate(var(--ci-ox), var(--ci-oy)); }}
      [data-testid="stChatInput"] textarea {{ background: var(--ci-bg)!important; border: var(--ci-bw) solid var(--ci-bc)!important; border-radius: var(--ci-r)!important; }}
      @media (max-width: 820px) {{ section.main > div.block-container > div:first-child > div:last-child {{ min-height: var(--chat-h-m); }} }}
    </style>
    """, unsafe_allow_html=True)

# --- HÀM CHÍNH MÀ app.py ĐANG IMPORT ---
def render_page(title: str = "TOMTRAN AGENT AI"):
    st.set_page_config(page_title=title, layout="wide")

    # đọc YAML cấu hình
    cfg = get_cfg()  # mặc định: ui_streamlit_theme.yaml cạnh file

    # inject CSS
    inject_css_from_cfg(cfg)

    # Khối điều khiển nhẹ (giống ChatGPT): trả về đủ 9 biến mà app.py cần dùng
    with st.sidebar:
        st.header("Sidebar")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.05)
        top_p = st.slider("Top-p", 0.1, 1.0, 1.0, 0.05)
        fallback_general = st.toggle("Fallback to general model", value=True)
        K = st.number_input("Retriever k", min_value=1, max_value=20, value=6, step=1)
        MIN_RELEVANCE = st.slider("Min relevance", 0.0, 1.0, 0.25, 0.01)
        debug_mode = st.toggle("Debug mode", value=False)
        show_system = st.toggle("Show system prompt", value=False)
        show_rag = st.toggle("Show RAG diagnostics", value=False)

    # Panel chat bên phải
    st.header("Chat")
    user_msg = st.chat_input("Enter your question here")

    # TRẢ VỀ đúng thứ tự mà app.py đang nhận:
    # user_msg, temperature, top_p, fallback_general, K, MIN_RELEVANCE, debug_mode, show_system, show_rag
    return user_msg, temperature, top_p, fallback_general, K, MIN_RELEVANCE, debug_mode, show_system, show_rag
