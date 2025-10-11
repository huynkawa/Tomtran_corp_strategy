# -*- coding: utf-8 -*-
"""
src/gpt_enhancer.py — GPT enhancer cho TABLE & TEXT
- TABLE:
    + mode="financial": BCTC (ép format CODE | NAME | NOTE | END | BEGIN)
    + mode="generic"  : bảng thường (cột linh hoạt, ngăn bằng '|')
    + Ưu tiên ẢNH nếu có; fallback về bản YAML-clean nếu GPT lỗi/format sai
    + Guardrail: kiểm định dạng '|', số cột tối thiểu, sanity check nhẹ cho BCTC
- TEXT:
    + Clean nhẹ văn bản thường (không đổi sang bảng/markdown)
- Backward-compat:
    + Cung cấp hàm enhance_with_gpt(...) để giữ tương thích ngược với code cũ
"""

from __future__ import annotations
import os, io, time, base64, json, re
from typing import Optional, List, Tuple, Dict, Any
from PIL import Image

# (tùy) nạp env key sớm (nếu bạn có src.env để load .env.active)
try:
    import src.env  # noqa
except Exception:
    pass

# OpenAI client (phiên bản v1+)
try:
    from openai import OpenAI
    _OPENAI_OK = True
except Exception:
    _OPENAI_OK = False


# ---------- Heuristic nhận dạng bảng BCTC ----------
_FIN_HINTS = [
    r"\bmã\s*số\b", r"\bchỉ\s*t[ií]êu\b",
    r"\bsố\s*cuối\s*năm\b", r"\bsố\s*đầu\s*năm\b",
    r"\bcode\b", r"\bend\b", r"\bbegin\b",
    r"\bassets\b", r"\bequity\b", r"\bliabilities\b"
]
_FIN_CODE_PAT = r"\b\d{2,3}(?:\.\d+)?\b"  # 100, 131, 131.1, 329.2,...

def detect_table_domain(clean_text: str) -> str:
    t = (clean_text or "").lower()
    if any(re.search(p, t, re.I) for p in _FIN_HINTS) and re.search(_FIN_CODE_PAT, t):
        return "financial"
    return "generic"


# ---------- Schema & Prompt builder ----------
def _schema_for_mode(mode: str, as_json: bool = False) -> Dict[str, Any]:
    mode = (mode or "financial").lower()
    if mode == "financial":
        if as_json:
            sys = (
                "Bạn là chuyên gia kiểm định bảng báo cáo tài chính (phi nhân thọ, VN). "
                "Đọc KỸ bảng trong ẢNH và đối chiếu với văn bản OCR đã làm sạch. "
                "Nếu mâu thuẫn, TIN ẢNH HƠN. KHÔNG BỊA. "
                "TRẢ VỀ JSON (list các hàng), mỗi hàng là object có khóa: "
                "CODE, NAME, NOTE, END, BEGIN. NOTE có thể rỗng. "
                "END/BEGIN dùng định dạng số kiểu VN dưới dạng chuỗi (ví dụ '1.234.567'). "
                "KHÔNG in thêm giải thích/markdown."
            )
        else:
            sys = (
                "Bạn là chuyên gia kiểm định bảng báo cáo tài chính (phi nhân thọ, VN). "
                "Đọc KỸ bảng trong ẢNH và đối chiếu với văn bản OCR đã làm sạch. "
                "Nếu mâu thuẫn, TIN ẢNH HƠN. KHÔNG BỊA. "
                "ĐẦU RA: TEXT THUẦN; mỗi dòng 1 hàng; cột theo thứ tự: "
                "CODE | NAME | NOTE | END | BEGIN. "
                "Nếu không có NOTE, để trống giữa hai dấu '|'. "
                "END/BEGIN định dạng số kiểu VN (1.234.567). "
                "Không in tiêu đề/markdown/giải thích."
            )
        return {"name": "financial", "min_cols": 4, "max_cols": 5, "sys": sys}
    else:
        if as_json:
            sys = (
                "Bạn là chuyên gia trích bảng trong ẢNH thành JSON. "
                "Đọc KỸ ẢNH và đối chiếu với văn bản OCR đã làm sạch. "
                "Nếu mâu thuẫn, TIN ẢNH HƠN. KHÔNG BỊA. "
                "TRẢ VỀ JSON: danh sách các hàng; mỗi hàng là list các cột theo thứ tự trái→phải. "
                "Không in thêm giải thích/markdown."
            )
        else:
            sys = (
                "Bạn là chuyên gia chuyển bảng trong ẢNH thành TEXT có cột. "
                "Đọc KỸ ẢNH và đối chiếu với văn bản OCR đã làm sạch. "
                "Nếu mâu thuẫn, TIN ẢNH HƠN. KHÔNG BỊA. "
                "ĐẦU RA: TEXT THUẦN; mỗi dòng 1 hàng; cột ngăn bằng '|', giữ thứ tự trái→phải. "
                "Không in tiêu đề/markdown/giải thích. "
                "Nếu số cột thay đổi giữa các hàng, vẫn in đúng theo quan sát."
            )
        return {"name": "generic", "min_cols": 3, "max_cols": None, "sys": sys}


# ---------- Utilities ----------
def _pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def _retry(fn, n=2, delay=1.0):
    err = None
    for i in range(n + 1):
        try:
            return fn()
        except Exception as e:
            err = e
            if i < n:
                time.sleep(delay * (2 ** i))
    raise err

def _postprocess_table_text(out: str, max_cols: Optional[int]) -> str:
    """Chuẩn hoá khoảng trắng quanh '|', cắt cột dư nếu max_cols được đặt."""
    lines = []
    for ln in (out or "").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        ln = re.sub(r"\s*\|\s*", " | ", ln)
        ln = re.sub(r"^\|\s*", "", ln)
        ln = re.sub(r"\s*\|$", "", ln)
        parts = [p.strip() for p in ln.split("|")]
        if max_cols and len(parts) > max_cols:
            parts = parts[:max_cols]
        ln = " | ".join(parts)
        lines.append(ln)
    return "\n".join(lines)

def _basic_guardrail_text(text: str, min_cols: int) -> bool:
    if not text:
        return False
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return False
    if all("|" not in ln for ln in lines):
        return False
    for ln in lines:
        if ln.count("|") + 1 < min_cols:
            return False
    return True

# parse TEXT dạng 'code | name | note | end | begin'
def _parse_financial_rows_text(text: str) -> List[Tuple[str, str, str, str, str]]:
    rows = []
    for ln in (text or "").splitlines():
        parts = [p.strip() for p in ln.split("|")]
        if len(parts) < 4:
            continue
        if len(parts) == 4:
            code, name, end, begin = parts
            note = ""
        else:
            code, name, note, end, begin = (parts + [""] * 5)[:5]
        rows.append((code, name, note, end, begin))
    return rows

_num_clean_re = re.compile(r"[^\d\.]")

def _to_number_like(s: str) -> Optional[int]:
    """Chuyển '1.234.567' → 1234567; nếu fail trả None."""
    if s is None:
        return None
    raw = _num_clean_re.sub("", s or "")
    if not raw:
        return None
    try:
        return int(raw.replace(".", ""))
    except Exception:
        return None

def _sanity_check_financial_text(out: str) -> bool:
    """Kiểm tra nhanh: có mã tổng quan trọng, và số END/BEGIN parse được."""
    rows = _parse_financial_rows_text(out)
    if not rows:
        return False
    codes = {r[0] for r in rows}
    if not codes.intersection({"100", "200", "270", "300", "400", "440"}):
        return False
    parsed_any = any((_to_number_like(end) is not None or _to_number_like(begin) is not None)
                     for _, _, _, end, begin in rows)
    return parsed_any

def _build_user_payload(table_text_cleaned: str, meta: Optional[dict]) -> List[dict]:
    user_text = (
        "VĂN BẢN ĐÃ LÀM SẠCH (BẢNG):\n"
        "-----BEGIN CLEANED TEXT-----\n"
        f"{table_text_cleaned}\n"
        "-----END CLEANED TEXT-----\n\n"
        "YÊU CẦU:\n"
        "- So và sửa theo ẢNH (nếu có) — ưu tiên ẢNH khi mâu thuẫn.\n"
        "- Trả KẾT QUẢ cuối cùng với mỗi dòng 1 hàng; cột ngăn bằng '|'.\n"
        "- Không thêm giải thích/markdown."
    )
    content = [{"type": "text", "text": user_text}]
    if meta:
        content.insert(0, {"type": "text", "text": "Meta (tham khảo, KHÔNG in ra):\n" + json.dumps(meta, ensure_ascii=False, indent=2)})
    return content


# ---------- API chính: TABLE ----------
def enhance_table_with_gpt(
    table_text_cleaned: str,
    image_pil: Optional[Image.Image] = None,
    meta: Optional[dict] = None,
    mode: Optional[str] = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_tokens: int = 3000,
    as_json: bool = False,
    log_diag: bool = True,
) -> str | List[Dict[str, Any]]:
    """
    Cross-check ảnh (nếu có) + text đã clean (YAML) → trả bảng theo schema mode.
    - mode=None → auto-detect (financial/generic)
    - as_json=True → trả JSON; False → TEXT.
    - Fallback: trả lại table_text_cleaned nếu GPT lỗi/format sai
    """
    if not _OPENAI_OK or not os.getenv("OPENAI_API_KEY"):
        if log_diag:
            print("⚠️ GPT skipped: OPENAI_API_KEY missing hoặc OpenAI lib không khả dụng.")
        return table_text_cleaned

    # auto detect nếu không truyền mode
    mode = mode or detect_table_domain(table_text_cleaned)
    schema = _schema_for_mode(mode, as_json=as_json)

    # build messages
    content = _build_user_payload(table_text_cleaned, meta)
    if image_pil is not None:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_pil_to_b64(image_pil)}"}})

    def _call():
        client = OpenAI()
        return client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": schema["sys"]},
                      {"role": "user", "content": content}],
            temperature=temperature,
            max_tokens=max_tokens,
        )

    try:
        resp = _retry(_call, n=2, delay=1.0)
        out = (resp.choices[0].message.content or "").strip()

        # log token nếu có
        if log_diag:
            used = getattr(getattr(resp, "usage", None), "total_tokens", None)
            print(f"🧠 GPT ok. mode={schema['name']} as_json={as_json} tokens≈{used if used is not None else '?'}")

        if as_json:
            # một số model có thể bọc ```json ...```
            raw = out.strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```json\s*|\s*```$", "", raw, flags=re.I | re.M)
            try:
                data = json.loads(raw)
            except Exception:
                if log_diag:
                    print("⚠️ JSON parse fail → fallback cleaned text.")
                return table_text_cleaned

            if schema["name"] == "financial":
                ok = isinstance(data, list) and all(isinstance(r, dict) for r in data)
                ok = ok and all(set(r.keys()) >= {"CODE", "NAME", "END", "BEGIN"} for r in data)
                if not ok:
                    if log_diag:
                        print("⚠️ JSON financial format invalid → fallback.")
                    return table_text_cleaned
            else:
                ok = isinstance(data, list) and all(isinstance(r, (list, tuple)) for r in data)
                if not ok:
                    if log_diag:
                        print("⚠️ JSON generic format invalid → fallback.")
                    return table_text_cleaned
            return data

        # TEXT mode: post-process + guardrail
        out = _postprocess_table_text(out, max_cols=schema.get("max_cols"))
        if not _basic_guardrail_text(out, schema["min_cols"]):
            if log_diag:
                print("⚠️ GPT output format invalid → fallback cleaned text.")
            return table_text_cleaned

        if schema["name"] == "financial" and not _sanity_check_financial_text(out):
            if log_diag:
                print("⚠️ Financial sanity check failed → fallback.")
            return table_text_cleaned

        return out

    except Exception as e:
        if log_diag:
            print(f"⚠️ OpenAI error → fallback cleaned text: {e}")
        return table_text_cleaned


# ---------- API phụ: TEXT (văn bản thường) ----------
def _enhance_plain_text_with_gpt(text_raw: str,
                                 meta: dict | None = None,
                                 model: str = "gpt-4o-mini",
                                 temperature: float = 0.2,
                                 max_tokens: int = 2000,
                                 log_diag: bool = True) -> str:
    """
    Dùng khi muốn clean VĂN BẢN THƯỜNG (không phải bảng).
    Làm sạch nhẹ: sửa lỗi OCR nhỏ, nối dòng gãy, giữ nguyên nội dung; không đổi sang bảng/markdown.
    """
    if not _OPENAI_OK or not os.getenv("OPENAI_API_KEY"):
        if log_diag:
            print("⚠️ GPT skipped (plain-text): OPENAI_API_KEY missing hoặc OpenAI lib không khả dụng.")
        return text_raw

    sys_prompt = (
        "Bạn là biên tập viên OCR. Hãy làm sạch đoạn văn sau: sửa lỗi chính tả OCR nhỏ, "
        "nối dòng gãy, giữ nguyên nội dung/ý, KHÔNG thêm/bớt, KHÔNG đổi thành bảng hay thêm markdown. "
        "Trả ra đúng văn bản sạch, thuần text."
    )
    content = []
    if meta:
        content.append({"type": "text", "text": "Meta (tham khảo, KHÔNG in ra):\n" + json.dumps(meta, ensure_ascii=False, indent=2)})
    content.append({"type": "text", "text": text_raw})

    def _call():
        client = OpenAI()
        return client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": sys_prompt},
                      {"role": "user", "content": content}],
            temperature=temperature,
            max_tokens=max_tokens,
        )

    try:
        resp = _retry(_call, n=2, delay=1.0)
        out = (resp.choices[0].message.content or "").strip()
        if log_diag:
            used = getattr(getattr(resp, "usage", None), "total_tokens", None)
            print(f"🧠 GPT ok (plain-text). tokens≈{used if used is not None else '?'}")
        return out or text_raw
    except Exception as e:
        if log_diag:
            print(f"⚠️ OpenAI error (plain-text) → fallback: {e}")
        return text_raw


# =======================
# Backward-compat shim(s)
# =======================
def enhance_with_gpt(
    text_raw: str,
    meta: dict | None = None,
    image_path: str | None = None,
    mode: str | None = None,
    **kwargs
) -> str:
    """
    Tương thích ngược với code cũ:
    - Nếu có ảnh hoặc nội dung trông như BẢNG → gọi enhance_table_with_gpt
    - Ngược lại → clean văn bản thường bằng _enhance_plain_text_with_gpt
    """
    # Heuristic: nếu có dấu '|' nhiều hoặc cues BCTC → coi như bảng
    looks_like_table = False
    t = (text_raw or "").lower()
    if ("|" in t and t.count("|") >= 2) or re.search(
        r"\bmã\s*số\b|\bsố\s*cuối\s*năm\b|\bsố\s*đầu\s*năm\b|\bcode\b", t, re.I
    ):
        looks_like_table = True

    pil = None
    if image_path and os.path.exists(image_path):
        try:
            pil = Image.open(image_path)
        except Exception:
            pil = None

    if looks_like_table or pil is not None:
        mode_eff = mode or ("financial" if looks_like_table else "generic")
        return enhance_table_with_gpt(
            table_text_cleaned=text_raw,
            image_pil=pil,
            meta=meta,
            mode=mode_eff,
            **kwargs
        )

    # Văn bản thường
    return _enhance_plain_text_with_gpt(
        text_raw=text_raw,
        meta=meta,
        **{k: v for k, v in kwargs.items() if k in {"model", "temperature", "max_tokens", "log_diag"}}
    )
