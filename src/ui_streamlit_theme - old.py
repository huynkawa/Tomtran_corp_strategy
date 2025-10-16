# src/ui_streamlit_theme.py
# =========================================================
# Nhiệm vụ:
# - Load YAML theme người dùng
# - Deep-merge với DEFAULT_CFG để luôn có đủ khóa
# - Tự sửa % nếu SIDEBAR + CHAT > 100
# - Schema có AXIS (mốc cục bộ) cho từng đối tượng
# =========================================================

from __future__ import annotations
from pathlib import Path
import yaml

# =======================
# DEFAULT CFG (đồng bộ với YAML)
# =======================
DEFAULT_CFG = {
    "META": {
        "VERSION": "2.0",
        "NOTES": "Default UI theme; override via YAML",
    },

    # ---------- TRỤC CĂN GIỮA TOÀN TRANG ----------
    "CENTER_AXIS": {
        "ENABLED": True,
        "MODE": "viewport",   # viewport | container
        "X_PCT": 50,          # ↑ phải | ↓ trái
        "Y_PCT": 50,          # ↑ dưới | ↓ trên
        "SNAP_TO_GRID": False,
        "GRID_STEP_PX": 8,
    },

    # ---------- MÀU SẮC CHUNG ----------
    "COLORS": {
        "BG": "#FFFFFF",      # nền tổng thể
        "TEXT": "#1F2328",    # màu chữ
        "CARD": "#FFFFFF",
        "LINE": "#E5E7EB",
        "PRIMARY": "#10A37F", # accent (đã map trong CSS)
    },

    # ---------- KIỂU CHỮ ----------
    "TYPOGRAPHY": {
        "FONT_FAMILY": "Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial",
        "BASE_SIZE_PX": 15,
        "TITLE_SIZE_PX": 20,
        "LINE_HEIGHT": 1.55,
    },

    # ---------- BỐ CỤC 2 CỘT + DIVIDER ----------
    "LAYOUT": {
        "MAIN_MAX_WIDTH_PCT": 100,
        "GUTTER_PX": 0,
        "SIDEBAR_WIDTH_PCT": 24,
        "CHAT_WIDTH_PCT": 76,     # ⚠ Tổng ≤100; dưới get_cfg có auto-fix
        "DIVIDER": {
            "SHOW": True,
            "COLOR": "#EEF2F7",
            "STYLE": "solid",
            "WIDTH_PX": 1,
            "OFFSET_X_PX": 0,
        },
    },

    # ---------- HIỆU ỨNG (để tránh KeyError) ----------
    "EFFECTS": {
        "SHADOW": False,
        "BLUR_PX": 0,
    },

    # ---------- ĐỐI TƯỢNG ----------
    "OBJECTS": {
        # ==== Sidebar (cột trái) ====
        "SIDEBAR": {
            "ENABLED": True,
            "OFFSET_X": "0px",
            "OFFSET_Y": "0px",
            "BG": "#0F172A",
            "TEXT": "#E2E8F0",
            "BORDER_COLOR": "#0F172A",
            "BORDER_PX": 0,
            "RADIUS_PX": 0,
            "SHADOW": False,
            "PADDING_PX": 14,

            # --- MỐC CỤC BỘ (kế thừa CENTER_AXIS nếu thiếu) ---
            "AXIS": {
                "INHERIT": True,         # true: kế thừa X/Y/MODE nếu thiếu
                "MODE": "inherit",       # inherit | viewport | container
                "X_PCT": 50,             # % tâm theo box sidebar
                "Y_PCT": 50,
                "OFFSET_X_PX": 0,        # dịch thêm từ tâm (px)
                "OFFSET_Y_PX": 0,
                "SCALE_X": 1.0,          # thu-phóng quanh tâm cục bộ
                "SCALE_Y": 1.0,
            },

            # --- DIVIDER cục bộ bám mốc Sidebar ---
            "DIVIDER": {
                "USE_LOCAL": True,       # true: dùng divider của sidebar (ưu tiên)
                "SHOW": True,
                "BASE_ON_AXIS": True,    # true: neo tại X_PCT của AXIS; false: neo theo cạnh
                "OFFSET_X_PX": 0,        # dịch từ mốc (px)
                "COLOR": "#EEF2F7",
                "STYLE": "solid",
                "WIDTH_PX": 1,
                "ORIGIN_EDGE": "right",  # chỉ dùng khi BASE_ON_AXIS=false
            },
        },

        # ==== Chat panel (cột phải) ====
        "CHAT_PANEL": {
            "ENABLED": True,
            "OFFSET_X": "0px",
            "OFFSET_Y": "0px",
            "BG": "#FFFFFF",
            "TEXT": "#1F2328",
            "BORDER_COLOR": "#FFFFFF",
            "BORDER_PX": 0,
            "RADIUS_PX": 0,
            "SHADOW": False,
            "PADDING_PX": 0,
            "CHAT_MAX_WIDTH_PX": 880,   # cột giữa giống ChatGPT
            "CHAT_HEIGHT_VH_DESKTOP": 72,
            "CHAT_HEIGHT_VH_MOBILE": 68,

            "AXIS": {
                "INHERIT": True,
                "MODE": "inherit",       # inherit | viewport | container
                "X_PCT": 50,
                "Y_PCT": 50,
                "OFFSET_X_PX": 0,
                "OFFSET_Y_PX": 0,
                "SCALE_X": 1.0,
                "SCALE_Y": 1.0,
            },
        },

        # ==== Ô nhập chat ====
        "CHAT_INPUT": {
            "ENABLED": True,
            "WIDTH_PCT": 100,
            "OFFSET_X": "0px",           # khi đã có chat
            "OFFSET_Y": "0px",
            "EMPTY_OFFSET_VH": -30,      # khi chưa có chat: kéo lên
            "BG": "#F7F7F8",
            "BORDER_COLOR": "#E5E7EB",
            "BORDER_PX": 1,
            "RADIUS_PX": 999,
            "STICKY_BOTTOM_PX": 20,

            "AXIS": {
                "INHERIT": True,
                "MODE": "inherit",
                "X_PCT": 50,
                "Y_PCT": 50,
                "OFFSET_X_PX": 0,
                "OFFSET_Y_PX": 0,
                "SCALE_X": 1.0,
                "SCALE_Y": 1.0,
            },
        },

        # ==== Bong bóng ====
        "BUBBLE_USER": {
            "ENABLED": True,
            "BG": "#E7F0FF",
            "TEXT": "#0B1520",
            "BORDER_COLOR": "#DCE6FF",
            "BORDER_PX": 1,
            "RADIUS_PX": 16,
        },
        "BUBBLE_BOT": {
            "ENABLED": True,
            "BG": "#FFFFFF",
            "TEXT": "#0B1520",
            "BORDER_COLOR": "#E5E7EB",
            "BORDER_PX": 1,
            "RADIUS_PX": 16,
        },
    },

    # ---------- THÀNH PHẦN BỔ TRỢ ----------
    "COMPONENTS": {
        "HIDE_STREAMLIT_CHROME": True,
        "EMPTY_STATE": {
            "SHOW": True,
            "TITLE_HTML": "Bạn đang làm về cái gì?",
            "SUB_HTML": "Hỏi bất kỳ điều gì…",
        },
        "SIDEBAR_CONTENT": {
            "SHOW": True,
            "SECTIONS": []
        }
    },
}

# =======================
# Deep merge
# =======================
def deep_update(base: dict, override: dict) -> dict:
    """Deep-merge override -> base (không phá cấu trúc)."""
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out

def _load_yaml_from_candidates(cands):
    for p in cands:
        p = Path(p)
        if p.is_file():
            try:
                with p.open("r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            except Exception:
                pass
    return {}

# =======================
# Public API
# =======================
def get_cfg(yaml_path: str | None = None) -> dict:
    """
    Trả về cấu hình UI đã deep-merge với DEFAULT_CFG.
    - Tự sửa % nếu SIDEBAR + CHAT > 100 (để không bể layout).
    - Chịu thiếu khóa an toàn.
    """
    if yaml_path:
        user_cfg = _load_yaml_from_candidates([yaml_path])
    else:
        candidates = [
            "configs/ui_streamlit_theme.yaml",
            ".streamlit/ui_streamlit_theme.yaml",
            "ui_streamlit_theme.yaml",
        ]
        user_cfg = _load_yaml_from_candidates(candidates)

    cfg = deep_update(DEFAULT_CFG, user_cfg or {})

    # Auto-fix: nếu tổng % > 100 → ép CHAT_WIDTH_PCT
    try:
        sb = int(cfg["LAYOUT"]["SIDEBAR_WIDTH_PCT"])
        ch = int(cfg["LAYOUT"]["CHAT_WIDTH_PCT"])
        if sb + ch > 100:
            cfg["LAYOUT"]["CHAT_WIDTH_PCT"] = max(0, 100 - sb)
    except Exception:
        pass

    return cfg
