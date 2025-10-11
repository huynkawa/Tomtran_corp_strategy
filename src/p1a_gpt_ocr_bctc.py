# -*- coding: utf-8 -*-
"""
src/gpt_enhancer.py — GPT enhancer cho TABLE (ưu tiên BCTC)

- Hàm public: enhance_table_with_gpt(...)
  * Nhận text bảng đã OCR + YAML-clean (table_text_cleaned)
  * Nhận ảnh PIL (image_pil) của trang để GPT đối chiếu (vision)
  * Chế độ: mode="financial" | "generic" | None (auto)
  * Trả về: string (bản bảng đã hiệu chỉnh bởi GPT) hoặc fallback bản đã clean nếu guardrail không đạt

- Guardrail chính:
  * Yêu cầu tối thiểu số cột theo schema (financial ≥ 5 cột: CODE|NAME|NOTE|END|BEGIN)
  * Đồng nhất số cột các dòng, không để dư thừa '|' ở cuối
  * Không bịa số nếu ảnh không có (nêu rõ yêu cầu trong prompt)

- Phụ thuộc:
  pip install openai pillow pyyaml

Lưu ý: Module này chỉ xử lý BẢNG. Phần văn bản đã được pipeline xử lý riêng.
"""
from __future__ import annotations
import io, base64, time, re, os
from typing import Optional, Dict, Any, List

from PIL import Image

# OpenAI SDK (>=1.0)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # sẽ xử lý fallback nếu SDK chưa có

# =========================
# ====== CONFIGS ==========
# =========================
FINANCIAL_MIN_COLS = 5     # CODE | NAME | NOTE | END | BEGIN
GENERIC_MIN_COLS   = 2

DEFAULT_MODEL      = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.0

# =========================
# ====== HELPERS ==========
# =========================

def pil_to_base64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _retry(fn, n=2, delay=1.0):
    last = None
    for i in range(n):
        try:
            return fn()
        except Exception as e:
            last = e
            time.sleep(delay)
    raise last


def _looks_like_table(text: str) -> bool:
    if not text:
        return False
    return ("|" in text) or bool(re.search(r"\b(mã\s*số|chi\s*tieu|số\s*cuối\s*năm|số\s*đầu\s*năm)\b", text, re.I))


def _basic_guardrail(out: str, min_cols: int) -> bool:
    if not out:
        return False
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    if not lines:
        return False
    # yêu cầu có dấu '|' và số cột tối thiểu
    col_counts: List[int] = []
    for ln in lines:
        if "|" not in ln:
            # cho phép header chú thích 1-2 dòng không có '|', nhưng phần lớn phải có
            continue
        # bỏ '|' đầu/cuối nếu có
        s = ln
        s = s[1:] if s.startswith("|") else s
        s = s[:-1] if s.endswith("|") else s
        cols = [c.strip() for c in s.split("|")]
        col_counts.append(len(cols))
        if len(cols) < min_cols:
            return False
    if not col_counts:
        return False
    # phần lớn dòng phải cùng số cột
    from statistics import mode as _mode
    try:
        majority = _mode(col_counts)
    except Exception:
        majority = max(set(col_counts), key=col_counts.count)
    ok_ratio = sum(1 for c in col_counts if c == majority) / max(1, len(col_counts))
    return ok_ratio >= 0.7


# =========================
# ====== PROMPTS ==========
# =========================

SYSTEM_FINANCIAL = (
    "Bạn là chuyên gia đọc BCTC tiếng Việt. Hãy trích xuất BẢNG tài chính chuẩn,"
    " dạng pipe với header: CODE | NAME | NOTE | END | BEGIN.\n"
    "- Chỉ dùng số có trong ẢNH; KHÔNG bịa số.\n"
    "- Giữ nguyên mã số (3 chữ số hoặc dạng phân cấp 1.1, 1.2...).\n"
    "- END = 'Số cuối năm', BEGIN = 'Số đầu năm' (nếu thiếu cột trong ảnh, để trống).\n"
    "- Hàng tổng hoặc mục lớn vẫn là một dòng với CODE tương ứng.\n"
    "- Không thêm cột ngoài schema, không chú thích dài dòng. Chỉ xuất bảng pipe."
)

SYSTEM_GENERIC = (
    "Bạn là chuyên gia chuyển bảng thành văn bản dạng pipe.\n"
    "- Chuẩn hóa bảng theo số cột nhất quán, tối thiểu 2 cột.\n"
    "- Dùng dữ liệu từ ẢNH làm nguồn chính; so khớp với bản text gợi ý.\n"
    "- Không thêm nhận xét hay giải thích. Chỉ xuất bảng pipe."
)


def _build_messages(model: str, image_b64: str, text_hint: str, meta: Dict[str, Any], mode: str):
    if mode == "financial":
        system = SYSTEM_FINANCIAL
        min_cols = FINANCIAL_MIN_COLS
        user_intro = (
            "ẢNH dưới đây là trang có BẢNG BCTC.\n"
            "Bản text gợi ý (đã OCR + YAML-clean) nằm sau đây để bạn đối chiếu:```\n"
            f"{text_hint}\n```.\n"
            "Hãy hiệu chỉnh dựa trên ảnh nếu thấy khác. Xuất DUY NHẤT bảng pipe."
        )
    else:
        system = SYSTEM_GENERIC
        min_cols = GENERIC_MIN_COLS
        user_intro = (
            "ẢNH dưới đây chứa bảng.\n"
            "Bản text gợi ý (đã OCR + YAML-clean):```\n"
            f"{text_hint}\n```.\n"
            "Hãy chuẩn hóa thành bảng pipe, dùng ảnh làm nguồn chính."
        )

    user_blocks: List[Dict[str, Any]] = [
        {"type": "text", "text": user_intro}
    ]
    # chèn ảnh
    user_blocks.append({
        "type": "input_image",
        "image_url": {
            "url": f"data:image/png;base64,{image_b64}",
        },
    })

    # chèn meta hint gọn
    hint_parts = []
    if meta:
        if meta.get("company_hint"): hint_parts.append(f"Company: {meta['company_hint']}")
        if meta.get("period_hint"):  hint_parts.append(f"Period: {meta['period_hint']}")
    if hint_parts:
        user_blocks.append({"type": "text", "text": " | ".join(hint_parts)})

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {"role": "user",   "content": user_blocks},
    ]
    return messages, min_cols


# =========================
# ====== MAIN API =========
# =========================

def enhance_table_with_gpt(
    table_text_cleaned: str,
    image_pil: Image.Image,
    meta: Optional[Dict[str, Any]] = None,
    *,
    mode: Optional[str] = "financial",
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    log_diag: bool = False,
) -> str:
    """
    Trả về: string bảng pipe đã hiệu chỉnh bởi GPT (nếu hợp lệ),
            nếu không hợp lệ → trả về table_text_cleaned (fallback)
    """
    # auto detect mode nếu None
    eff_mode = mode
    if eff_mode not in ("financial", "generic"):
        eff_mode = "financial" if _looks_like_table(table_text_cleaned) else "generic"

    # Chuẩn bị ảnh
    if not isinstance(image_pil, Image.Image):
        raise ValueError("image_pil phải là PIL.Image.Image")
    b64 = pil_to_base64_png(image_pil.convert("RGB"))

    # Nếu không có SDK thì fallback
    if OpenAI is None:
        if log_diag:
            print("⚠️ OpenAI SDK chưa cài. Fallback về YAML-clean.")
        return table_text_cleaned

    client = OpenAI()

    # Tạo messages
    messages, min_cols = _build_messages(model, b64, table_text_cleaned, meta or {}, eff_mode)

    def _call():
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=messages,
        )
        return resp

    try:
        resp = _retry(_call, n=2, delay=1.0)
        out = (resp.choices[0].message.content or "").strip()
        if log_diag:
            print("[GPT RAW OUTPUT]\n" + out[:1200] + ("..." if len(out) > 1200 else ""))
        if not _basic_guardrail(out, min_cols=min_cols):
            if log_diag:
                print("⚠️ Guardrail không đạt → fallback YAML-clean.")
            return table_text_cleaned
        # dọn '|'
        clean_lines = []
        for ln in out.splitlines():
            s = ln.strip()
            if not s:
                continue
            # bỏ '|' thừa đầu/cuối + chuẩn hóa khoảng trắng
            if s.startswith("|"): s = s[1:]
            if s.endswith("|"):   s = s[:-1]
            s = "|".join([c.strip() for c in s.split("|")])
            clean_lines.append(s)
        return "\n".join(clean_lines)
    except Exception as e:
        if log_diag:
            print(f"⚠️ OpenAI error → fallback YAML-clean: {e}")
        return table_text_cleaned
