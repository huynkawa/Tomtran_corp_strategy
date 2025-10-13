# src/ui_streamlit_layout.py
import streamlit as st
from src.ui_streamlit_theme import get_cfg

def _val(d, k, default=None):
    return d.get(k, default) if isinstance(d, dict) else default

def _inject_css(cfg: dict):
    axis, colors, typo, layout, obj, comp = (
        cfg.get("CENTER_AXIS", {}), cfg.get("COLORS", {}),
        cfg.get("TYPOGRAPHY", {}), cfg.get("LAYOUT", {}),
        cfg.get("OBJECTS", {}),  cfg.get("COMPONENTS", {})
    )
    sb, ch, ci = obj.get("SIDEBAR", {}), obj.get("CHAT_PANEL", {}), obj.get("CHAT_INPUT", {})
    div = layout.get("DIVIDER", {})

    center_x = f"{_val(axis,'X_PCT',50)}vw"
    center_y = f"{_val(axis,'Y_PCT',50)}vh"
    maxw     = _val(ch, "CHAT_MAX_WIDTH_PX", 760)
    sticky   = _val(ci, "STICKY_BOTTOM_PX", 20)

    hide_chrome = comp.get("HIDE_STREAMLIT_CHROME", True)
    chrome_css = "#MainMenu, header[data-testid='stHeader'], footer, .stDeployButton {display:none!important;}" if hide_chrome else ""

    st.markdown(f"""
    <style id="axis-driven-ui">
      {chrome_css}
      :root {{
        --bg:{_val(colors,'BG','#fff')}; --text:{_val(colors,'TEXT','#111')};
        --card:{_val(colors,'CARD','#fff')}; --line:{_val(colors,'LINE','#e5e7eb')};
        --base:{_val(typo,'BASE_SIZE_PX',15)}px; --title:{_val(typo,'TITLE_SIZE_PX',20)}px; --font:{_val(typo,'FONT_FAMILY','system-ui')};

        --main-max:{_val(layout,'MAIN_MAX_WIDTH_PCT',96)}vw; --gutter:{_val(layout,'GUTTER_PX',0)}px;
        --sidebar-w:{_val(layout,'SIDEBAR_WIDTH_PCT',18)}%; --chat-w:{_val(layout,'CHAT_WIDTH_PCT',82)}%;

        --divider-show:{'block' if _val(div,'SHOW',True) else 'none'};
        --divider-color:{_val(div,'COLOR','#e5e7eb')};
        --divider-style:{_val(div,'STYLE','solid')};
        --divider-width:{_val(div,'WIDTH_PX',1)}px; --divider-ox:{_val(div,'OFFSET_X_PX',0)}px;

        --center-x:{center_x}; --center-y:{center_y};

        --sb-ox:{_val(sb,'OFFSET_X','0px')}; --sb-oy:{_val(sb,'OFFSET_Y','0px')};
        --sb-bg:{_val(sb,'BG','#0F172A')}; --sb-text:{_val(sb,'TEXT','#E2E8F0')};
        --sb-bc:{_val(sb,'BORDER_COLOR','#0F172A')}; --sb-bw:{_val(sb,'BORDER_PX',0)}px; --sb-r:{_val(sb,'RADIUS_PX',12)}px;
        --sb-pad:{_val(sb,'PADDING_PX',8)}px; --sb-shadow:{'0 8px 20px rgba(0,0,0,.25)' if _val(sb,'SHADOW',True) else 'none'};

        --ch-ox:{_val(ch,'OFFSET_X','0px')}; --ch-oy:{_val(ch,'OFFSET_Y','0px')};
        --ch-bg:{_val(ch,'BG','#fff')}; --ch-text:{_val(ch,'TEXT','#111')};
        --ch-bc:{_val(ch,'BORDER_COLOR','#fff')}; --ch-bw:{_val(ch,'BORDER_PX',0)}px; --ch-r:{_val(ch,'RADIUS_PX',0)}px; --ch-pad:{_val(ch,'PADDING_PX',0)}px;
        --chat-max:{maxw}px; --chat-h-d:{_val(ch,'CHAT_HEIGHT_VH_DESKTOP',72)}vh; --chat-h-m:{_val(ch,'CHAT_HEIGHT_VH_MOBILE',68)}vh;

        --ci-w:{_val(ci,'WIDTH_PCT',100)}%; --ci-ox:{_val(ci,'OFFSET_X','0px')}; --ci-oy:{_val(ci,'OFFSET_Y','0px')};
        --ci-bg:{_val(ci,'BG','#F7F7F8')}; --ci-bc:{_val(ci,'BORDER_COLOR','#E5E7EB')}; --ci-bw:{_val(ci,'BORDER_PX',1)}px; --ci-r:{_val(ci,'RADIUS_PX',999)}px;
        --ci-sticky:{sticky}px;

        --user-bg:{_val(obj.get('BUBBLE_USER',{}),'BG','#E5F1FF')};
        --user-text:{_val(obj.get('BUBBLE_USER',{}),'TEXT','#111')};
        --user-bc:{_val(obj.get('BUBBLE_USER',{}),'BORDER_COLOR','#BFDBFE')};
        --user-bw:{_val(obj.get('BUBBLE_USER',{}),'BORDER_PX',1)}px; --user-r:{_val(obj.get('BUBBLE_USER',{}),'RADIUS_PX',18)}px;

        --bot-bg:{_val(obj.get('BUBBLE_BOT',{}),'BG','#fff')};
        --bot-text:{_val(obj.get('BUBBLE_BOT',{}),'TEXT','#111')};
        --bot-bc:{_val(obj.get('BUBBLE_BOT',{}),'BORDER_COLOR','#E5E7EB')};
        --bot-bw:{_val(obj.get('BUBBLE_BOT',{}),'BORDER_PX',1)}px; --bot-r:{_val(obj.get('BUBBLE_BOT',{}),'RADIUS_PX',18)}px;
      }}

      body,.stApp {{ background: var(--bg)!important; color: var(--text)!important; font-family: var(--font)!important; font-size: var(--base); }}

      section.main > div.block-container {{
        max-width: var(--main-max)!important; padding-left: var(--gutter)!important; padding-right: var(--gutter)!important;
      }}
      section.main > div.block-container > div:first-child {{
        display:grid; grid-template-columns:minmax(0,var(--sidebar-w)) minmax(0,var(--chat-w));
        gap:0; position:relative;
      }}
      section.main > div.block-container > div:first-child::before {{
        content:""; display: var(--divider-show);
        position:absolute; top:0; bottom:0; left: calc(var(--sidebar-w) + var(--divider-ox));
        width: var(--divider-width); background: var(--divider-color);
        border-left: var(--divider-width) var(--divider-style) var(--divider-color); z-index:9;
      }}

      /* SIDEBAR theo TRỤC */
      [data-testid="stSidebar"] > div {{
        background: var(--sb-bg)!important; color: var(--sb-text)!important;
        border: var(--sb-bw) solid var(--sb-bc); border-radius: var(--sb-r);
        box-shadow: var(--sb-shadow); padding: var(--sb-pad);
        transform: translate(calc(var(--center-x) + var(--sb-ox) - 50%), calc(var(--center-y) + var(--sb-oy) - 50%));
      }}

      /* CHAT PANEL theo TRỤC + cột giữa */
      section.main > div.block-container > div:first-child > div:last-child {{
        background: var(--ch-bg); color: var(--ch-text);
        border: var(--ch-bw) solid var(--ch-bc); border-radius: var(--ch-r);
        padding: var(--ch-pad); min-height: var(--chat-h-d);
        transform: translate(calc(var(--center-x) + var(--ch-ox) - 50%), calc(var(--center-y) + var(--ch-oy) - 50%));
      }}
      .chat-center {{ display:flex; justify-content:center; min-height: 60vh; }}
      .chat-col    {{ width:100%; max-width: var(--chat-max); margin:0 auto; padding:24px 16px calc(var(--ci-sticky) + 66px) 16px; }}

      /* Bubbles */
      .stChatMessage > div {{
        border: var(--bot-bw) solid var(--bot-bc)!important;
        border-radius: var(--bot-r)!important; background: var(--bot-bg)!important; color: var(--bot-text)!important; padding:14px 16px!important;
      }}
      .stChatMessage[data-testid="stChatMessage-User"] > div {{
        background: var(--user-bg)!important; color: var(--user-text)!important;
        border: var(--user-bw) solid var(--user-bc)!important; border-radius: var(--user-r)!important;
      }}

      /* Chat input: sticky theo YAML */
      [data-testid="stChatInput"] {{
        position: {('fixed' if sticky and sticky>0 else 'static')} !important;
        left: 50%; transform: translateX({('-50%' if sticky and sticky>0 else '0')});
        bottom: {sticky}px; width: min(var(--ci-w), var(--chat-max)); margin: 0 auto; z-index: 50;
      }}
      [data-testid="stChatInput"] textarea {{
        background: var(--ci-bg)!important; border: var(--ci-bw) solid var(--ci-bc)!important; border-radius: var(--ci-r)!important;
        box-shadow: 0 2px 12px rgba(0,0,0,.06);
      }}
      [data-testid="stChatInput"] label {{ display:none!important; }}

      @media (max-width: 820px) {{
        section.main > div.block-container > div:first-child > div:last-child {{ min-height: var(--chat-h-m); }}
      }}
    </style>
    """, unsafe_allow_html=True)

def _render_sidebar(cfg: dict):
    comp = cfg.get("COMPONENTS", {})
    sb_cfg = comp.get("SIDEBAR_CONTENT", {})
    if not sb_cfg.get("SHOW", True):
        return
    st.markdown(f"### {sb_cfg.get('HEADER','Sidebar')}")
    for item in sb_cfg.get("ITEMS", []):
        t = item.get("type")
        if t == "slider":
            st.slider(item.get("label",""), item.get("min",0.0), item.get("max",1.0),
                      item.get("value",0.5), item.get("step",0.01))
        elif t == "toggle":
            st.toggle(item.get("label",""), value=item.get("value",False))
        elif t == "number":
            st.number_input(item.get("label",""), min_value=item.get("min",0), max_value=item.get("max",100),
                            value=item.get("value",0), step=item.get("step",1))

def render_page(title: str = "TOMTRAN AGENT AI"):
    st.set_page_config(page_title=title, layout="wide")
    cfg = get_cfg("configs/ui_streamlit_theme.yaml")
    _inject_css(cfg)

    col_sidebar, col_chat = st.columns([1, 4], gap="small")

    with col_sidebar:
        if _val(cfg.get("OBJECTS",{}).get("SIDEBAR",{}), "ENABLED", True):
            _render_sidebar(cfg)

    with col_chat:
        st.markdown('<div class="chat-center"><div class="chat-col">', unsafe_allow_html=True)
        empty_cfg = cfg.get("COMPONENTS", {}).get("EMPTY_STATE", {})
        if empty_cfg.get("SHOW", True) and not st.session_state.get("messages"):
            st.markdown(f"<h2 style='text-align:center;margin:48px 0 24px 0;'>{empty_cfg.get('TITLE_HTML','')}</h2>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align:center;opacity:.65;'>{empty_cfg.get('SUB_HTML','')}</p>", unsafe_allow_html=True)
        for role, content in st.session_state.get("messages", []):
            with st.chat_message("user" if role=="user" else "assistant"):
                st.markdown(content)
        st.markdown('</div></div>', unsafe_allow_html=True)

    # Chat input theo YAML
    if _val(cfg.get("OBJECTS",{}).get("CHAT_INPUT",{}), "ENABLED", True):
        user_msg = st.chat_input("Enter your question here")
    else:
        user_msg = None

    if user_msg:
        st.session_state.setdefault("messages", []).append(("user", user_msg))
        st.rerun()

    # Trả về chữ ký cũ (nếu app.py đang dùng)
    temperature = 0.3; top_p = 1.0; fallback_general = True
    K = 6; MIN_RELEVANCE = 0.25; debug_mode=False; show_system=False; show_rag=False
    return user_msg, temperature, top_p, fallback_general, K, MIN_RELEVANCE, debug_mode, show_system, show_rag
