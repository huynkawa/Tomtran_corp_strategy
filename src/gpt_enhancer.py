# -*- coding: utf-8 -*-
"""
src/gpt_enhancer.py — GPT enhancer cho TABLE & TEXT (generic-first, TSV)
- TABLE:
    + Mặc định: generic (mọi loại bảng chiến lược/Excel, không ép schema tài chính)
    + Financial chỉ khi meta.class == "financial" hoặc mode="financial"
    + Xuất TSV (tab) chuẩn; nếu model trả '|', sẽ chuyển về TAB
    + Guardrail nhẹ: min_cols, định dạng dòng; sanity-check tài chính chỉ khi financial_strict=True
- TEXT:
    + Clean nhẹ văn bản thường (không đổi sang bảng/markdown)
- Backward-compat:
    + Giữ enhance_with_gpt(...) để tương thích code runner cũ
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

# OpenAI client (v1+)
try:
    from openai import OpenAI
    _OPENAI_OK = True
except Exception:
    _OPENAI_OK = False


# ---------- Heuristic nhận dạng domain ----------
_FIN_HINTS = [
    r"\bmã\s*số\b", r"\bchỉ\s*t[ií]êu\b",
    r"\bsố\s*cuối\s*năm\b", r"\bsố\s*đầu\s*năm\b",
    r"\bcode\b", r"\bend\b", r"\bbegin\b",
    r"\bassets\b", r"\bequity\b", r"\bliabilities\b"
]
_FIN_CODE_PAT = r"\b\d{2,3}(?:\.\d+)?\b"  # 100, 131, 131.1, 329.2,...

_STRATEGY_HINTS = [
    r"\bkpi\b", r"\bbsc\b", r"\bobjective\b", r"\bindicator\b",
    r"\btarget\b", r"\baction\b", r"\bretention\b", r"\bauthority\b",
    r"\bmetric\b", r"\bscorecard\b", r"\bclient\b", r"\bcustomer\b",
    r"\bunit\b", r"\bgoal\b", r"\bmeasure\b", r"\buw\b", r"\bunderr?writing\b"
]

def _looks_financial(text: str) -> bool:
    t = (text or "").lower()
    return any(re.search(p, t, re.I) for p in _FIN_HINTS) and re.search(_FIN_CODE_PAT, t)

def detect_table_domain(clean_text: str, meta: Optional[dict]=None) -> str:
    """
    Mặc định: 'generic'. Chỉ trả 'financial' khi meta.class == 'financial'
    hoặc văn bản có tín hiệu tài chính MẠNH.
    """
    if meta and str(meta.get("class", "")).lower() == "financial":
        return "financial"
    if _looks_financial(clean_text):
        return "financial"
    # Bất kỳ tín hiệu chiến lược nào cũng ưu tiên generic (không ép schema)
    t = (clean_text or "").lower()
    if any(re.search(p, t, re.I) for p in _STRATEGY_HINTS):
        return "generic"
    return "generic"


# ---------- Schema & Prompt builder ----------
def _schema_for_mode(mode: str, as_json: bool = False, sep: str = "\t") -> Dict[str, Any]:
    """
    sep: ký tự phân cột yêu cầu trong output (mặc định TAB).
    """
    mode = (mode or "generic").lower()
    if mode == "financial":
        if as_json:
            sys = (
                "Bạn là chuyên gia trích bảng TÀI CHÍNH từ ảnh/văn bản OCR.\n"
                "TRẢ VỀ JSON: danh sách các hàng; mỗi hàng là object có khóa: "
                "CODE, NAME, NOTE (có thể rỗng), END, BEGIN.\n"
                "Không thêm giải thích/markdown."
            )
        else:
            sep_name = "TAB" if sep == "\t" else sep
            sys = (
                "Bạn là chuyên gia trích bảng TÀI CHÍNH từ ảnh/văn bản OCR.\n"
                f"ĐẦU RA: TEXT THUẦN; mỗi dòng 1 hàng; cột ngăn bằng '{sep_name}'.\n"
                "Thứ tự cột: CODE, NAME, NOTE (có thể rỗng), END, BEGIN.\n"
                "Không in tiêu đề/markdown/giải thích."
            )
        return {"name": "financial", "min_cols": 4, "max_cols": 5, "sys": sys, "sep": sep}
    else:
        if as_json:
            sys = (
                "Bạn là chuyên gia trích bảng từ ảnh/văn bản OCR (không bắt buộc tài chính).\n"
                "TRẢ VỀ JSON: danh sách các hàng; mỗi hàng là list các cột trái→phải.\n"
                "Không thêm giải thích/markdown."
            )
        else:
            sep_name = "TAB" if sep == "\t" else sep
            sys = (
                "Bạn là chuyên gia chuyển bảng từ ảnh/văn bản OCR thành TEXT có cột.\n"
                f"ĐẦU RA: TEXT THUẦN; mỗi dòng 1 hàng; cột ngăn bằng '{sep_name}', giữ thứ tự trái→phải.\n"
                "Không in tiêu đề/markdown/giải thích.\n"
                "Nếu số cột thay đổi giữa các hàng, vẫn in đúng theo quan sát."
            )
        return {"name": "generic", "min_cols": 2, "max_cols": None, "sys": sys, "sep": sep}


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

def _normalize_to_sep(out: str, sep: str) -> str:
    """
    Chuẩn hóa output về cùng 1 dấu phân cột (sep).
    - Chấp nhận model trả bằng '|' hoặc TAB; sẽ convert về 'sep'
    """
    if not out:
        return ""
    lines = []
    for ln in out.splitlines():
        s = ln.strip()
        if not s:
            continue
        # Thử tách theo TAB trước
        if "\t" in s and sep == "\t":
            parts = [p.strip() for p in s.split("\t")]
        else:
            # nếu có '|', tách theo '|'
            if "|" in s and (sep == "\t" or sep == "|"):
                parts = [p.strip() for p in re.split(r"\s*\|\s*", s)]
            else:
                # fallback: coi như 1 cột
                parts = [s.strip()]
        # Ghép theo sep
        if sep == "\t":
            s_norm = "\t".join(parts)
        elif sep == "|":
            s_norm = " | ".join(parts)
        else:
            s_norm = sep.join(parts)
        lines.append(s_norm)
    return "\n".join(lines)

def _basic_guardrail_text(text: str, min_cols: int, sep: str) -> bool:
    if not text:
        return False
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return False
    # Chấp nhận nếu có ít nhất một trong hai dấu (sep hoặc '|') để nhận diện cột
    col_sep = sep
    def count_cols(ln: str) -> int:
        if col_sep in ln:
            return ln.count(col_sep) + 1
        if "|" in ln:
            return ln.count("|") + 1
        return 1
    ok = any(((col_sep in ln) or ("|" in ln)) for ln in lines)
    if not ok:
        return False
    # kiểm số cột tối thiểu
    return all(count_cols(ln) >= min_cols for ln in lines)


# ---------- Financial helpers (optional/lenient) ----------
def _parse_financial_rows_text(text: str, sep: str) -> List[Tuple[str, str, str, str, str]]:
    rows = []
    for ln in (text or "").splitlines():
        parts = [p.strip() for p in (ln.split(sep) if sep in ln else ln.split("|"))]
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
    if s is None:
        return None
    raw = _num_clean_re.sub("", s or "")
    if not raw:
        return None
    try:
        return int(raw.replace(".", ""))
    except Exception:
        return None

def _sanity_check_financial_text(out: str, sep: str) -> bool:
    rows = _parse_financial_rows_text(out, sep)
    if not rows:
        return False
    # chỉ kiểm tra rất nhẹ: có parse được ít nhất 1 số
    parsed_any = any((_to_number_like(end) is not None or _to_number_like(begin) is not None)
                     for _, _, _, end, begin in rows)
    return parsed_any


# ---------- User payload builder ----------
def _build_user_payload(table_text_cleaned: str, meta: Optional[dict]) -> List[dict]:
    user_text = (
        "VĂN BẢN ĐÃ LÀM SẠCH (BẢNG):\n"
        "-----BEGIN CLEANED TEXT-----\n"
        f"{table_text_cleaned}\n"
        "-----END CLEANED TEXT-----\n\n"
        "YÊU CẦU:\n"
        "- Nếu có ẢNH thì dùng ảnh để đối chiếu, ưu tiên ẢNH khi mâu thuẫn.\n"
        "- Trả KẾT QUẢ cuối cùng, mỗi dòng 1 hàng, đúng số cột; KHÔNG thêm giải thích/markdown."
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
    *,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_tokens: int = 3000,
    as_json: bool = False,
    financial_strict: bool = False,
    sep: str = "\t",
    log_diag: bool = True,
) -> str | List[Dict[str, Any]]:
    """
    Trích bảng theo generic-first, xuất TSV (sep='\t' mặc định).
    - mode=None → detect theo meta/text (ưu tiên generic)
    - as_json=True → trả JSON; False → TEXT.
    - financial_strict: chỉ áp khi mode='financial' (mặc định False).
    - Fallback: trả lại table_text_cleaned nếu GPT lỗi/format sai.
    """
    if not _OPENAI_OK or not os.getenv("OPENAI_API_KEY"):
        if log_diag:
            print("⚠️ GPT skipped: OPENAI_API_KEY missing hoặc OpenAI lib không khả dụng.")
        return table_text_cleaned

    # domain detect
    mode = (mode or detect_table_domain(table_text_cleaned, meta)).lower()
    schema = _schema_for_mode(mode, as_json=as_json, sep=sep)
    # override min/max columns từ meta (nếu có)
    min_cols = int((meta or {}).get("table_min_cols", schema.get("min_cols", 2)) or 2)
    max_cols = (meta or {}).get("table_max_cols", schema.get("max_cols"))
    schema["min_cols"] = min_cols
    schema["max_cols"] = max_cols

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

        # TEXT mode → normalize về sep + guardrail
        out = _normalize_to_sep(out, sep=schema["sep"])
        if not _basic_guardrail_text(out, schema["min_cols"], sep=schema["sep"]):
            if log_diag:
                print("⚠️ GPT output format invalid → fallback cleaned text.")
            return table_text_cleaned

        # Sanity tài chính chỉ khi bật financial_strict
        if schema["name"] == "financial" and financial_strict:
            if not _sanity_check_financial_text(out, sep=schema["sep"]):
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
    Clean văn bản thường (OCR): sửa lỗi nhỏ, nối dòng gãy, giữ nguyên nội dung.
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
    image: str | None = None,  # path hoặc PIL sẽ được auto mở ở runner
    mode: str | None = None,
    **kwargs
) -> str:
    """
    Tương thích với runner:
    - Nếu có dấu hiệu bảng (nhiều '|' hoặc TAB) hoặc có ảnh → enhance_table_with_gpt
    - Ngược lại → _enhance_plain_text_with_gpt
    Hỗ trợ tham số mới:
      * sep="\t" (mặc định TSV)
      * financial_strict=False
    """
    looks_like_table = False
    t = (text_raw or "").lower()
    if ("\t" in t) or ("|" in t and t.count("|") >= 2):
        looks_like_table = True

    # runner gửi image=path; cố gắng mở
    pil = None
    if image and isinstance(image, str) and os.path.exists(image):
        try:
            pil = Image.open(image)
        except Exception:
            pil = None

    if looks_like_table or pil is not None:
        # default generic-first, TSV
        sep = kwargs.pop("sep", "\t")
        financial_strict = kwargs.pop("financial_strict", False)
        return enhance_table_with_gpt(
            table_text_cleaned=text_raw,
            image_pil=pil,
            meta=meta,
            mode=mode,  # None → auto detect (ưu tiên generic)
            sep=sep,
            financial_strict=financial_strict,
            **{k: v for k, v in kwargs.items() if k in {"model", "temperature", "max_tokens", "as_json", "log_diag"}}
        )

    # Văn bản thường: tôn trọng meta.enable_paragraph_gpt (mặc định False)
    enable_text = bool((meta or {}).get("enable_paragraph_gpt", False))
    if not enable_text:
        return text_raw  # giữ nguyên text (đã có sanitizer ở runner)
    return _enhance_plain_text_with_gpt(
        text_raw=text_raw,
        meta=meta,
        **{k: v for k, v in kwargs.items() if k in {"model", "temperature", "max_tokens", "log_diag"}}
    )
