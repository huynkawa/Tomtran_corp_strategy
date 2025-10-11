# src/ui_streamlit_theme.py
from dataclasses import dataclass
from pathlib import Path
import yaml
import streamlit as st

@dataclass
class UiTheme:
    # ========= LOGO & TIÊU ĐỀ =========
    TITLE_FONT_SIZE_REM: float = 1.90          # ↑ tăng = chữ tiêu đề to hơn
    LOGO_SIZE_PX: int = 140                    # ↑ tăng = logo to hơn
    LOGO_OFFSET_REM: float = 0.0               # ↑ dương = header/logo hạ xuống; ↑ âm = kéo lên

    LOGO_PATH: str = "assets/Dancing Chatbot.mp4"  # đường dẫn logo (ảnh/video)
    LOGO_IS_VIDEO: bool = True                     # True: nhúng <video>, False: <img>
    SHOW_TITLE: bool = True                        # ẩn/hiện chữ tiêu đề cạnh logo

    # Tuỳ biến khung logo (dùng trong CSS ở layout)
    LOGO_BORDER_RADIUS_PX: int = 50               # 50≈tròn; 0=vuông
    LOGO_OBJECT_FIT: str = "cover"                # "cover" = cắt cho đầy khung; "contain" = không cắt
    LOGO_SHADOW: bool = True                      # True=đổ bóng nhẹ, False=tắt bóng

    # ========= BANNER DƯỚI HEADER =========
    HEADER_DIVIDER_MODE: str = "line"             # "text" = banner chữ + 2 vạch, "line" = chỉ kẻ
    HEADER_DIVIDER_TEXT: str = ""                 # nội dung chữ ở giữa (để trống = không hiện chữ)

    HEADER_DIVIDER_TOP_REM: float = 0.6           # ↑ tăng = banner hạ thấp xuống
    HEADER_DIVIDER_BOTTOM_REM: float = 0.35       # ↑ tăng = nới khoảng trống bên dưới
    HEADER_DIVIDER_WIDTH_PCT: int = 82            # ↑ tăng = cụm banner rộng hơn (theo % trang)
    HEADER_DIVIDER_THICK_PX: int = 1              # ↑ tăng = 2 vạch dày hơn
    HEADER_DIVIDER_STYLE: str = "solid"           # solid | dashed | dotted
    HEADER_DIVIDER_COLOR: str = "#2E2E2E"         # màu 2 vạch
    # Cho line cũ: vẫn tôn trọng cờ SHOW_HEADER_DIVIDER nếu có trong YAML
    SHOW_HEADER_DIVIDER: bool = True              # True=hiện line khi MODE != "text" hoặc TEXT trống

    # Kiểu chữ banner
    HEADER_DIVIDER_TEXT_COLOR: str = "#bfbfbf"    # ↑ gần #fff = chữ sáng hơn
    HEADER_DIVIDER_TEXT_SIZE_REM: float = 0.95    # ↑ tăng = chữ to hơn
    HEADER_DIVIDER_TEXT_WEIGHT: int = 600         # 400=thường, 700=đậm

    # ========= KHOẢNG CÁCH & BỐ CỤC =========
    HEADER_TO_CHAT_GAP_REM: float = -2.0          # ↑ âm = kéo khung chat sát logo

    CHAT_HEIGHT_VH_EMPTY: int = 32                # chiều cao khung chat khi trống
    CHAT_HEIGHT_VH_NONEMPTY: int = 62             # chiều cao khung chat khi có tin

    CHAT_PAD_V_REM: float = 1.10                  # padding trên/dưới của khung chat
    CHAT_PAD_H_REM: float = 1.60                  # padding trái/phải của khung chat

    CHAT_WIDTH_DESKTOP_PX: int = 820              # bề rộng chat desktop (px)
    CHAT_WIDTH_TABLET_PX: int  = 680              # bề rộng chat tablet (px)
    CHAT_WIDTH_MOBILE_PCT: int = 96               # bề rộng chat mobile (%)

    INPUT_MARGIN_TOP_REM: float = -1.5            # đẩy input lên/xuống
    LEFT_SIDEBAR_WIDTH_PX: int = 290              # chiều rộng sidebar trái

    # ========= STRAP & FOOTER =========
    SHOW_STRAP: bool = True
    STRAP_ICON: str = "🧭"
    STRAP_TEXT: str = "The Best Assistant of Tom Tran"

    SHOW_FOOTER: bool = True
    FOOTER_TITLE: str = "© TOMTRAN Chatbot"
    FOOTER_TAGLINE: str = "Strategic insight & business intelligence."
    FOOTER_COPYRIGHT: str = "© 2025 Tom Tran Corporation. All rights reserved."
    FOOTER_STRAP_TOP_MARG_REM: float = 0.10       # khoảng cách strap với phần trên
    FOOTER_BRAND_TOP_MARG_REM: float = 0.35       # khoảng cách footer với phần trên


def _read_yaml_to_theme() -> UiTheme:
    """Đọc YAML, nếu có key trùng thuộc tính thì gán đè vào UiTheme."""
    theme = UiTheme()
    cfg = Path("configs/ui_streamlit_theme.yaml")
    if cfg.exists():
        try:
            data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
            for k, v in data.items():
                if hasattr(theme, k):
                    setattr(theme, k, v)
        except Exception as e:
            st.sidebar.warning(f"⚠️ ui_streamlit_theme.yaml lỗi: {e}")
    return theme


@st.cache_resource
def _cached_theme(yaml_mtime: float) -> UiTheme:
    """Cache theo mtime của YAML để mỗi lần save là tự reload."""
    return _read_yaml_to_theme()


def load_theme() -> UiTheme:
    cfg = Path("configs/ui_streamlit_theme.yaml")
    mtime = cfg.stat().st_mtime if cfg.exists() else 0.0
    return _cached_theme(mtime)
