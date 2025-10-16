# src/ui_streamlit_layout.py
import streamlit as st
from src.ui_streamlit_theme import get_cfg
# Ép sidebar mở sẵn mỗi lần load (Streamlit hay nhớ trạng thái lần trước)
try:
    st.set_page_config(layout="wide", initial_sidebar_state="expanded")
except Exception:
    pass

def _resolve_axis(global_ca: dict, obj_cfg: dict):
    axis = (obj_cfg or {}).get("AXIS", {}) or {}
    inherit = axis.get("INHERIT", True)
    mode = axis.get("MODE", "inherit")
    if inherit or str(mode).lower() == "inherit":
        mode = global_ca.get("MODE", "viewport")
    xp = axis.get("X_PCT", global_ca.get("X_PCT", 50))
    yp = axis.get("Y_PCT", global_ca.get("Y_PCT", 50))
    ofx = axis.get("OFFSET_X_PX", 0)
    ofy = axis.get("OFFSET_Y_PX", 0)
    sx = axis.get("SCALE_X", 1.0)
    sy = axis.get("SCALE_Y", 1.0)
    return {"mode": str(mode).lower(), "xp": float(xp), "yp": float(yp), "ofx": float(ofx), "ofy": float(ofy), "sx": float(sx), "sy": float(sy)}

def inject_css_from_cfg(cfg: dict, *, empty_state: bool = False):
    co = cfg.get("COLORS", {})
    tp = cfg.get("TYPOGRAPHY", {})
    ly = cfg.get("LAYOUT", {})
    ca = cfg.get("CENTER_AXIS", {})
    layers = cfg.get("LAYERS", {}) or {}

    obj = cfg.get("OBJECTS", {})
    sb = obj.get("SIDEBAR", {})
    ch = obj.get("CHAT_PANEL", {})
    ci = obj.get("CHAT_INPUT", {})

    ax_sb = _resolve_axis(ca, sb)
    ax_ch = _resolve_axis(ca, ch)
    ax_in = _resolve_axis(ca, ci)

    # ----- z-index lấy từ YAML -----
    z_base    = int(layers.get("BASE", 0))
    z_header  = int(layers.get("HEADER", 80))
    z_chat    = int(ch.get("Z_INDEX", layers.get("CHAT", 100)))
    z_sidebar = int(sb.get("Z_INDEX", layers.get("SIDEBAR", 200)))
    # divider: ưu tiên override ở SIDEBAR.DIVIDER.Z_INDEX
    sb_div = (sb.get("DIVIDER") or {})
    z_divider = int(sb_div.get("Z_INDEX", layers.get("DIVIDER", 210)))
    z_input  = int(ci.get("Z_INDEX", layers.get("INPUT", 900)))
    z_ovl    = int(layers.get("OVERLAY", 950))
    z_toggle = int(layers.get("TOGGLE", 10000))

    # Divider global (fallback)
    divider_g = (ly.get("DIVIDER") or {})
    divider_show_g = bool(divider_g.get("SHOW", False))
    divider_color_g = divider_g.get("COLOR", "#E5E7EB")
    divider_style_g = divider_g.get("STYLE", "solid")
    divider_w_g = int(divider_g.get("WIDTH_PX", 1))

    # Divider local options
    sb_div_use = bool(sb_div.get("USE_LOCAL", False))
    sb_div_show = bool(sb_div.get("SHOW", False))
    sb_div_base_on_axis = bool(sb_div.get("BASE_ON_AXIS", True))
    sb_div_color = sb_div.get("COLOR", "#EEF2F7")
    sb_div_style = sb_div.get("STYLE", "solid")
    sb_div_w = int(sb_div.get("WIDTH_PX", 1))
    sb_div_offx = float(sb_div.get("OFFSET_X_PX", 0))
    sb_div_edge = str(sb_div.get("ORIGIN_EDGE", "right")).lower()

    # Empty state input offset
    input_offset_vh = f"{ci.get('EMPTY_OFFSET_VH', -30)}vh" if empty_state else None

    # Ẩn chrome: chỉ ẩn menu & footer (giữ header để luôn có nút toggle)
    hide_chrome_css = "#MainMenu, footer {visibility: hidden;}" \
        if (cfg.get("COMPONENTS", {}).get("HIDE_STREAMLIT_CHROME")) else ""

    divider_global_css = (
        f"border-right: {divider_w_g}px {divider_style_g} {divider_color_g};"
        if (divider_show_g and not (sb_div_use and sb_div_show)) else ""
    )

    if sb_div_use and sb_div_show:
        if sb_div_base_on_axis:
            sidebar_local_divider_css = (
                "section[data-testid='stSidebar']::after{"
                f"content:'';position:absolute;top:0;bottom:0;width:{sb_div_w}px;"
                f"background:{sb_div_color};left:calc(var(--sb-xp) + var(--sb-ofx));"
                "transform:translateX(-50%);}"
            )
        else:
            # --- BASE_ON_AXIS == False: neo theo cạnh ---
            if sb_div_edge == "left":
                pos = f"left:{int(sb_div_offx)}px;"
            else:
                pos = f"right:{abs(int(sb_div_offx))}px;"

            # GHÉP CHUỖI AN TOÀN: f-string chỉ cho phần cần chèn biến; dấu '}' kết thúc khối CSS nằm ở string thường
            sidebar_local_divider_css = (
                "section[data-testid='stSidebar']::after{"
                f"content:'';position:absolute;top:0;bottom:0;width:{sb_div_w}px;"
                f"background:{sb_div_color};{pos}transform:none;"
                "}"
            )

    else:
        sidebar_local_divider_css = ""

    empty_input_center_css = (
        f".tt-input-wrap{{ position: fixed; top: {input_offset_vh}; left: 50%; "
        "transform: translate(-50%, -50%) scale(var(--in-sx), var(--in-sy)); "
        "transform-origin: var(--in-xp) var(--in-yp); }}"
        if empty_state else ""
    )

    primary = co.get("PRIMARY", "#10A37F")

    css = f"""
    <style>
      :root {{
        --bg: {co.get('BG', '#FFFFFF')};
        --text: {co.get('TEXT', '#1F2328')};
        --line: {co.get('LINE', '#E5E7EB')};
        --primary: {primary};
        --font: {tp.get('FONT_FAMILY', 'Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial')};
        --base: {tp.get('BASE_SIZE_PX', 15)}px;

        --chat-max: {ch.get('CHAT_MAX_WIDTH_PX', 880)}px;

        /* AXIS biến cho từng đối tượng */
        --sb-xp: {ax_sb['xp']}%; --sb-yp: {ax_sb['yp']}%; --sb-ofx: {ax_sb['ofx']}px; --sb-ofy: {ax_sb['ofy']}px; --sb-sx: {ax_sb['sx']}; --sb-sy: {ax_sb['sy']};
        --ch-xp: {ax_ch['xp']}%; --ch-yp: {ax_ch['yp']}%; --ch-ofx: {ax_ch['ofx']}px; --ch-ofy: {ax_ch['ofy']}px; --ch-sx: {ax_ch['sx']}; --ch-sy: {ax_ch['sy']};
        --in-xp: {ax_in['xp']}%; --in-yp: {ax_in['yp']}%; --in-ofx: {ax_in['ofx']}px; --in-ofy: {ax_in['ofy']}px; --in-sx: {ax_in['sx']}; --in-sy: {ax_in['sy']};

        /* Z-INDEX từ YAML (LAYERS + override) */
        --z-base: {z_base};
        --z-header: {z_header};
        --z-chat: {z_chat};
        --z-sidebar: {z_sidebar};
        --z-divider: {z_divider};
        --z-input: {z_input};
        --z-overlay: {z_ovl};
        --z-toggle: {z_toggle};
      }}

      .stApp, .stApp > div, .block-container {{
        background: var(--bg) !important;
        color: var(--text);
        font-family: var(--font);
        font-size: var(--base);
      }}

      /* Sidebar */
      section[data-testid="stSidebar"] {{
        background: {sb.get('BG', '#0F172A')} !important;
        color: {sb.get('TEXT', '#E2E8F0')} !important;
        position: relative;
        z-index: var(--z-sidebar);
        transform-origin: var(--sb-xp) var(--sb-yp);
        transform: translate(var(--sb-ofx), var(--sb-ofy)) scale(var(--sb-sx), var(--sb-sy));
        margin-left: {sb.get('OFFSET_X', '0px')};
        margin-top: {sb.get('OFFSET_Y', '0px')};
      }}
      section[data-testid="stSidebar"] > div:first-child {{
        padding: {sb.get('PADDING_PX', 14)}px;
        {divider_global_css}
      }}

      /* Divider local của sidebar */
      section[data-testid="stSidebar"]::after {{
        z-index: var(--z-divider);
      }}
      {sidebar_local_divider_css}

      /* Khung chat giữa */
      .tt-chat-inner {{
        width: min(100%, var(--chat-max));
        margin-left: auto; margin-right: auto;
        position: relative;
        z-index: var(--z-chat);
        transform-origin: var(--ch-xp) var(--ch-yp);
        transform: translate(var(--ch-ofx), var(--ch-ofy)) scale(var(--ch-sx), var(--ch-sy));
      }}

      /* Empty state */
      .tt-empty {{ min-height: 40vh; display: grid; place-items: center; text-align: center; margin-top: 10vh; }}

      /* Input */
      .tt-input-wrap {{
        position: sticky; bottom: {ci.get('STICKY_BOTTOM_PX', 20)}px; margin-top: 16px;
        z-index: var(--z-input);
        transform-origin: var(--in-xp) var(--in-yp);
        transform: translate(var(--in-ofx), var(--in-ofy)) scale(var(--in-sx), var(--in-sy));
      }}
      .tt-input {{
        background: {ci.get('BG', '#F7F7F8')};
        border: {ci.get('BORDER_PX', 1)}px solid {ci.get('BORDER_COLOR', '#E5E7EB')};
        border-radius: {ci.get('RADIUS_PX', 999)}px; padding: 8px 12px;
      }}

      /* Accent cho slider/toggle */
      .stSlider [data-baseweb="slider"] div[role="slider"] {{ background-color: var(--primary) !important; }}
      .stSlider [data-baseweb="slider"] div[aria-hidden="true"] {{ background-color: var(--primary) !important; }}
      .stSwitch [data-testid="stSwitch"] div[role="switch"][aria-checked="true"] {{
        background-color: var(--primary) !important; border-color: var(--primary) !important;
      }}

      /* Nút toggle sidebar luôn hiện & trên cùng */
      [data-testid="collapsedControl"] {{
        position: fixed; left: 12px; top: 12px; z-index: var(--z-toggle);
        visibility: visible !important; opacity: 1 !important;
      }}

      /* Ẩn chrome (giữ header) */
      {hide_chrome_css}

      /* Input giữa viewport khi empty-state */
      {empty_input_center_css}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def render_sidebar_from_yaml(cfg: dict):
    comp = (cfg.get("COMPONENTS", {}).get("SIDEBAR_CONTENT") or cfg.get("SIDEBAR_CONTENT", {}))
    if not comp or not comp.get("SHOW"):
        return
    sections = comp.get("SECTIONS", [])
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("current_title", None)

    def _new_chat():
        st.session_state["messages"] = []
        st.session_state["current_title"] = None

    for item in sections:
        t = (item.get("type") or "").lower()
        if t == "new_chat_button":
            st.button(item.get("label","Đoạn chat mới"), use_container_width=True, on_click=_new_chat)
        elif t == "history":
            st.caption(item.get("label","Lịch sử đoạn chat"))
            if st.session_state["chat_history"]:
                for i, title in enumerate(st.session_state["chat_history"], 1):
                    if st.button(f"{i}. {title}", key=f"hist_{i}", use_container_width=True):
                        st.session_state["current_title"] = title
            else:
                st.write("— chưa có đoạn chat —")
        elif t == "spacer":
            st.write("")
        elif t == "slider":
            st.session_state[item["label"]] = st.slider(item["label"], item["min"], item["max"], item["value"], item["step"])
        elif t == "toggle":
            st.session_state[item["label"]] = st.toggle(item["label"], value=item["value"])
        elif t == "number":
            st.session_state[item["label"]] = st.number_input(item["label"], min_value=item["min"], max_value=item["max"], value=item["value"], step=item["step"])
        else:
            st.write(f"⚠️ Unknown sidebar item type: {t}")

def render_page():
    cfg = get_cfg()
    empty_state = cfg.get("COMPONENTS", {}).get("EMPTY_STATE", {}).get("SHOW", False) and not st.session_state.get("messages")
    inject_css_from_cfg(cfg, empty_state=empty_state)

    with st.sidebar:
        render_sidebar_from_yaml(cfg)
        temperature   = st.session_state.get("Temperature", 0.3)
        top_p         = st.session_state.get("Top-p", 1.0)
        fallback_general = st.session_state.get("Fallback", True)
        K             = st.session_state.get("Retriever k", 6)
        MIN_RELEVANCE = st.session_state.get("Min relevance", 0.25)
        debug_mode    = st.session_state.get("Debug mode", False)
        show_system   = st.session_state.get("Show system prompt", False)
        show_rag      = st.session_state.get("Show RAG diagnostics", False)

    if empty_state:
        es = cfg.get("COMPONENTS", {}).get("EMPTY_STATE", {})
        st.markdown(
            f"""
            <div class="tt-chat-inner">
              <div class="tt-empty">
                <div>
                  <h2 style="margin:0 0 8px 0;">{es.get("TITLE_HTML","")}</h2>
                  <div style="opacity:.7;">{es.get("SUB_HTML","")}</div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with st.container():
        st.markdown('<div class="tt-chat-inner"><div class="tt-input-wrap"><div class="tt-input">', unsafe_allow_html=True)
        user_msg = st.chat_input("Enter your question here")
        st.markdown('</div></div></div>', unsafe_allow_html=True)

    return user_msg, temperature, top_p, fallback_general, K, MIN_RELEVANCE, debug_mode, show_system, show_rag
