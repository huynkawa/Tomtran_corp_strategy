# -*- coding: utf-8 -*-
"""
p1a_gpt_ocr_bctc.py — GPT enhancer cho TABLE (ưu tiên BCTC, đọc ẢNH + đối chiếu PIPE)
- Hàm public: enhance_table_with_gpt(...)
  * Nhận:  table_text_cleaned (PIPE var-cols do code dựng) + image_pil (ROI hoặc cả trang)
  * GPT đọc ẢNH, đối chiếu với PIPE, sửa số liệu theo ẢNH (không bịa), giữ định dạng PIPE
  * Chế độ: mode="financial" | "generic" | "auto"
  * Trả về: PIPE đã hiệu chỉnh; nếu guardrail không đạt → fallback về table_text_cleaned

Guardrail:
  - Bắt buộc có ký tự '|' (định dạng pipe)
  - Số cột đồng nhất theo đa số; với financial: mỗi dòng phải >= 2 cột & đa số dòng >= 5 cột (CODE|NAME|NOTE|END|BEGIN)
  - Tự chuẩn hoá số kiểu Việt (dấu chấm ngăn nghìn), giữ âm (cả dạng (123) → -123)
  - Nếu output GPT không hợp lệ / lỗi mạng → fallback

Phụ thuộc:
  pip install openai pillow
"""

from __future__ import annotations
import io, base64, time, re
from typing import Optional, Dict, Any, List

from PIL import Image

# OpenAI SDK (>=1.0)
try:
    from openai import OpenAI
    _HAS_OPENAI = True
except Exception:
    OpenAI = None
    _HAS_OPENAI = False


# =========================
# ====== CONFIGS ==========
# =========================
FINANCIAL_MIN_COLS = 5          # CODE | NAME | NOTE | END | BEGIN (khuyến nghị)
GENERIC_MIN_COLS   = 2

DEFAULT_MODEL       = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.0

# prompt system (ngắn gọn, nhấn mạnh ưu tiên ẢNH & pipe)
SYSTEM_FINANCIAL = (
    "Bạn là chuyên gia đọc bảng BCTC từ ảnh scan. ƯU TIÊN SỐ LIỆU TỪ ẢNH, KHÔNG BỊA. "
    "Giữ dạng bảng PIPE (các cột ngăn bởi ' | '). Nếu có mâu thuẫn giữa ảnh và văn bản gợi ý, "
    "hãy chỉnh theo ẢNH. Chuẩn hoá số kiểu Việt (dấu chấm ngăn nghìn). Nếu không chắc, để trống ô."
)

SYSTEM_GENERIC = (
    "Bạn là chuyên gia đọc bảng từ ảnh scan. ƯU TIÊN SỐ LIỆU TỪ ẢNH, KHÔNG BỊA. "
    "Giữ dạng bảng PIPE (các cột ngăn bởi ' | '). Chuẩn hoá số kiểu Việt nếu có. "
    "Nếu không chắc, để trống ô."
)

# user yêu cầu output đúng block để dễ cắt
USER_INSTR = (
    "NHIỆM VỤ:\n"
    "1) Đọc ẢNH bảng.\n"
    "2) Đối chiếu với bảng PIPE gợi ý (từ OCR code) và SỬA SỐ LIỆU theo ẢNH.\n"
    "3) Giữ định dạng PIPE (các cột ngăn bởi ' | '), không thêm chú thích ngoài bảng.\n"
    "4) Nếu không chắc một ô số → để trống.\n"
    "5) Xuất KẾT QUẢ trong block:\n"
    "<<<PIPE>>>\n"
    "<bảng pipe cuối cùng>\n"
    "<<<END>>>"
)


# =========================
# ====== HELPERS ==========
# =========================

# [MOD] 1) Resize ảnh về cạnh dài ~1600px để tránh input quá nặng
def _pil_to_base64_png(img: Image.Image) -> str:
    # shrink longest side to 1600px to keep tokens/latency bounded
    MAX_SIDE = 1600
    w, h = img.size
    if max(w, h) > MAX_SIDE:
        scale = MAX_SIDE / float(max(w, h))
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _retry(fn, n=2, delay=1.0):
    last = None
    for _ in range(n):
        try:
            return fn()
        except Exception as e:
            last = e
            time.sleep(delay)
    if last:
        raise last


def _looks_like_table(text: str) -> bool:
    if not text:
        return False
    if "|" in text:
        return True
    return bool(re.search(r"\b(mã\s*số|chi\s*t[iê]u|số\s*cuối\s*năm|số\s*đầu\s*năm)\b", text, re.I))


def _normalize_vn_amount(s: str) -> str:
    """
    Chuẩn hoá số kiểu Việt, giữ âm (kẻ cả (123) → -123).
    Không cố gắng chuyển text thuần sang số.
    """
    t = (s or "").strip()
    if not t:
        return ""
    neg = False
    if t.startswith("(") and t.endswith(")"):
        neg = True
    # chỉ giữ ký tự số, dấu, khoảng trắng
    t = re.sub(r"[^\d\.,\-\s]", "", t)
    t = t.replace(",", ".")
    t = re.sub(r"\s+", "", t)
    t = re.sub(r"\.{2,}", ".", t).strip(".")

    # nếu là chuỗi số dài -> nhóm 3
    digits = re.sub(r"[^\d\-]", "", t)
    if digits in ("", "-", "--"):
        return s.strip()
    sign = ""
    if digits.startswith("-"):
        sign, digits = "-", digits[1:]

    if len(digits) <= 3:
        out = digits
    else:
        parts = []
        while digits:
            parts.append(digits[-3:])
            digits = digits[:-3]
        out = ".".join(reversed(parts))

    if sign or neg:
        out = "-" + out

    # [MOD] 5) tidy up các trường hợp '-0' → '0', và không trả rỗng vô nghĩa
    if out == "-0":
        out = "0"
    if out == "":
        return s.strip()

    return out


def _post_clean_pipe(out_pipe: str, financial_mode: bool) -> str:
    """
    - Bỏ '|' đầu/cuối, chuẩn hoá khoảng trắng mỗi cell.
    - Chuẩn hoá số (cell có vẻ là số lớn) theo kiểu Việt.
    """
    lines = [ln for ln in (out_pipe or "").splitlines() if ln.strip()]
    if not lines:
        return ""

    cleaned = []
    for idx, ln in enumerate(lines):
        s = ln.strip()
        if s.startswith("|"):
            s = s[1:]
        if s.endswith("|"):
            s = s[:-1]
        cells = [c.strip() for c in s.split("|")]

        # Chuẩn hoá số cho mọi cell có vẻ là số
        new_cells = []
        for c in cells:
            if re.search(r"\d{1,3}([.,]\d{3}){1,}", c) or re.fullmatch(r"[\-\(\)\d\.\, ]{3,}", c):
                norm = _normalize_vn_amount(c)
                new_cells.append(norm if norm else c.strip())
            else:
                new_cells.append(c.strip())
        cleaned.append(" | ".join(new_cells))
    return "\n".join(cleaned)


def _guardrail_pipe(out_pipe: str, min_cols_required: int, financial_mode: bool) -> bool:
    """
    - Bắt buộc có '|' trên đa số dòng
    - Số cột đồng nhất theo đa số (mode)
    - Với financial_mode: đa số dòng >= 5 cột
    """
    if not out_pipe or "|" not in out_pipe:
        return False

    lines = [ln for ln in out_pipe.splitlines() if ln.strip()]
    if len(lines) < 2:
        return False

    col_counts: List[int] = []
    lines_with_pipe = 0
    for ln in lines:
        if "|" not in ln:
            continue
        lines_with_pipe += 1
        s = ln.strip()
        if s.startswith("|"):
            s = s[1:]
        if s.endswith("|"):
            s = s[:-1]
        cols = [c.strip() for c in s.split("|")]
        col_counts.append(len(cols))

    if lines_with_pipe < max(2, int(0.7 * len(lines))):
        return False

    if not col_counts:
        return False

    # đa số cột
    try:
        from statistics import mode as _mode
        maj = _mode(col_counts)
    except Exception:
        maj = max(set(col_counts), key=col_counts.count)

    if maj < max(2, min_cols_required):
        return False

    # [MOD] 3) Nếu financial: ưu tiên >=5 cột, nhưng chấp nhận 4 cột khi rất nhất quán
    if financial_mode:
        enough5 = sum(1 for c in col_counts if c >= FINANCIAL_MIN_COLS)
        if enough5 < int(0.6 * len(col_counts)):
            # chấp nhận 4 cột nếu đa số tuyệt đối và có logic END/BEGIN (phần này sẽ do caller map)
            enough4 = sum(1 for c in col_counts if c >= 4)
            if enough4 < int(0.8 * len(col_counts)):
                return False

    return True


def _extract_pipe_block(text: str) -> str:
    """Cắt phần giữa <<<PIPE>>> ... <<<END>>> nếu có; nếu không thì dùng toàn bộ text."""
    if not text:
        return ""
    m = re.search(r"<<<PIPE>>>\s*(.*?)\s*<<<END>>>", text, flags=re.S)
    return (m.group(1) if m else text).strip()


def _build_messages(image_b64: str, pipe_hint: str, mode: str, meta: Dict[str, Any]) -> tuple[list, int, bool]:
    """Trả về (messages, min_cols_required, financial_mode_flag)"""
    financial_mode = (mode or "").lower() == "financial"
    system = SYSTEM_FINANCIAL if financial_mode else SYSTEM_GENERIC
    min_cols = FINANCIAL_MIN_COLS if financial_mode else GENERIC_MIN_COLS

    # meta hint gọn
    hint_meta = []
    if meta:
        if meta.get("company_hint"):
            hint_meta.append(f"Company: {meta['company_hint']}")
        if meta.get("period_hint"):
            hint_meta.append(f"Period: {meta['period_hint']}")
    meta_text = (" | ".join(hint_meta)) if hint_meta else ""

    user_blocks: List[Dict[str, Any]] = [
        {"type": "text", "text": USER_INSTR},
        {"type": "text", "text": "Đây là ẢNH bảng cần làm chuẩn:"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
        {"type": "text", "text": "Đây là bảng PIPE do code dựng (để cross-check & sửa):\n" + pipe_hint}
    ]
    if meta_text:
        user_blocks.append({"type": "text", "text": meta_text})

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {"role": "user",   "content": user_blocks},
    ]
    return messages, min_cols, financial_mode


# [MOD] 2) Cắt gọn pipe hint nếu quá dài để tiết kiệm context
def _truncate_pipe_hint(pipe: str, max_lines: int = 220) -> str:
    lines = [ln for ln in (pipe or "").splitlines() if ln.strip()]
    if len(lines) <= max_lines:
        return pipe
    head = lines[:20]                 # giữ header + vài dòng đầu
    tail = lines[-(max_lines - 20):]  # ưu tiên phần cuối (thường chứa tổng)
    return "\n".join(head + ["..."] + tail)


# =========================
# ====== MAIN API =========
# =========================

def enhance_table_with_gpt(
    table_text_cleaned: str,
    image_pil: Image.Image,
    meta: Optional[Dict[str, Any]] = None,
    *,
    mode: str = "financial",               # "financial" | "generic" | "auto"
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    log_diag: bool = False,
) -> str:
    """
    GPT đọc ẢNH + PIPE → cross-check → trả ra PIPE đã sửa.
    Nếu thất bại/không hợp lệ → trả về table_text_cleaned (fallback).
    """

    # 1) Kiểm tra đầu vào
    pipe_hint = (table_text_cleaned or "").strip()
    pipe_hint = _truncate_pipe_hint(pipe_hint, max_lines=220)  # [MOD] 2) truncate hint nếu quá dài

    if not isinstance(image_pil, Image.Image):
        # Có thể gọi nhầm không đưa ảnh -> fallback
        if log_diag:
            print("⚠️ enhance_table_with_gpt: image_pil không phải PIL.Image → fallback PIPE gợi ý.")
        return pipe_hint

    # 2) Quyết định mode
    eff_mode = (mode or "auto").lower()
    if eff_mode not in ("financial", "generic"):
        # auto: nếu có dấu hiệu code + nhiều số tiền → financial
        has_code = bool(re.search(r"(?m)^\s*(\d{3}(?:\.\d+)?)\b", pipe_hint))
        many_money = sum(1 for ln in pipe_hint.splitlines() if len(re.findall(r"\d{1,3}(?:[.,]\d{3}){1,}", ln)) >= 2) >= 4
        eff_mode = "financial" if (has_code and many_money) else "generic"

    # 3) Nếu chưa có SDK → fallback
    if not _HAS_OPENAI:
        if log_diag:
            print("⚠️ OpenAI SDK chưa sẵn sàng → fallback PIPE gợi ý.")
        return pipe_hint

    # 4) Chuẩn bị ảnh & messages
    b64 = _pil_to_base64_png(image_pil.convert("RGB"))
    messages, min_cols_required, financial_mode = _build_messages(b64, pipe_hint, eff_mode, meta or {})

    if log_diag:
        print(f"[GPT CALL] model={model} temp={temperature} mode={eff_mode} "
              f"| min_cols={min_cols_required} | financial={financial_mode}")
        print("[PIPE HINT PREVIEW]\n" + "\n".join(pipe_hint.splitlines()[:6]) + ("\n..." if len(pipe_hint.splitlines()) > 6 else ""))

    # 5) Gọi GPT (có retry)
    # [MOD] 4) Fail-safe khi init client lỗi (thiếu API key, network, ...)
    try:
        client = OpenAI()  # sẽ đọc OPENAI_API_KEY từ env
    except Exception as e:
        if log_diag:
            print(f"⚠️ OpenAI init error → fallback PIPE gợi ý: {e}")
        return pipe_hint

    def _call():
        return client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=messages,
        )

    try:
        resp = _retry(_call, n=2, delay=1.0)
        raw = (resp.choices[0].message.content or "").strip()
        if log_diag:
            print("[GPT RAW OUTPUT]\n" + raw[:1600] + ("..." if len(raw) > 1600 else ""))

        # 6) Cắt block PIPE & hậu xử lý
        pipe_out = _extract_pipe_block(raw)
        pipe_out = _post_clean_pipe(pipe_out, financial_mode=financial_mode)

        # 7) Guardrail
        if not _guardrail_pipe(pipe_out, min_cols_required=min_cols_required, financial_mode=financial_mode):
            if log_diag:
                print("⚠️ Guardrail không đạt → fallback PIPE gợi ý.")
            return pipe_hint

        return pipe_out

    except Exception as e:
        if log_diag:
            print(f"⚠️ OpenAI error → fallback PIPE gợi ý: {e}")
        return pipe_hint
    
# ====== COMPAT WRAPPER (giữ API cũ enhance_with_gpt) ======
def enhance_with_gpt(
    text_raw: str,
    meta: Optional[Dict[str, Any]] = None,
    image_path: Optional[str] = None,
    *,
    mode: str = "financial",
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    log_diag: bool = False,
) -> str:
    """
    Wrapper để tương thích code cũ:
    - Nhận text_raw (kỳ vọng là PIPE nếu là bảng), meta (dict), image_path (đường dẫn ảnh)
    - Mở ảnh → PIL.Image rồi gọi enhance_table_with_gpt(...)
    - Nếu không có ảnh hoặc không mở được ảnh → fallback text_raw
    - Nếu text_raw không phải PIPE (không có '|') → để nhẹ nhàng trả lại text_raw
    """
    try:
        # Nếu không có '|' → khả năng không phải bảng PIPE → không ép GPT
        if "|" not in (text_raw or ""):
            if log_diag:
                print("ℹ️ enhance_with_gpt: input không có '|' → trả lại text_raw (không gọi GPT).")
            return (text_raw or "")

        if image_path:
            try:
                img = Image.open(image_path).convert("RGB")
            except Exception as e:
                if log_diag:
                    print(f"⚠️ Không mở được ảnh ({image_path}): {e} → fallback text_raw")
                return (text_raw or "")
        else:
            if log_diag:
                print("⚠️ Không có image_path → fallback text_raw")
            return (text_raw or "")

        return enhance_table_with_gpt(
            table_text_cleaned=(text_raw or ""),
            image_pil=img,
            meta=(meta or {}),
            mode=mode,
            model=model,
            temperature=temperature,
            log_diag=log_diag,
        )
    except Exception as e:
        if log_diag:
            print(f"⚠️ Wrapper enhance_with_gpt error → fallback: {e}")
        return (text_raw or "")

