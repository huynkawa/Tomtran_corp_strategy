# -*- coding: utf-8 -*-
"""
src/c2_ocr_scan_bctc_outputs.py — OCR từ ảnh prelight (ưu tiên _bin.png) → TXT + META

Mục tiêu:
- Đọc ảnh đã qua prelight (deskew/crop/binarize) để OCR bền vững hơn so với render PDF trực tiếp.
- Xuất đúng 2 file/trang:
    <base>_page{n}_text.txt
    <base>_page{n}_meta.json
- Giữ logic nhận diện: company / unit (VND & multiplier) / period (as_of / year_ended / quarter_ended) /
  report title / statement / language
- Reflow bằng TSV (bbox) để hạn chế gãy dòng vùng “Chỉ tiêu”.

Chạy mẫu:
  python -m src.p1a_clean10_ocr_bctc --start 8 --end 8

Yêu cầu:
  pip install pdf2image pillow opencv-python-headless numpy pytesseract
  + Đã cài Tesseract (tesseract.exe có trong PATH hoặc đặt env TESSERACT_CMD)
"""
from __future__ import annotations
import os, re, glob, json, argparse, hashlib, shutil
from typing import Optional, Tuple, Dict, List

import numpy as np
import cv2
from PIL import Image
import pytesseract
from pytesseract import Output as TessOutput
# ==== YAML CONFIG (validator & text-clean) ====
import yaml

# Cho phép override qua ENV, nếu không thì dùng đường dẫn bạn cung cấp:
YAML_TEXT_PATH  = os.getenv(
    "P1A_YAML_TEXT",
    r"D:\1.TLAT\3. ChatBot_project\1_Insurance_Strategy\configs\c2_ocr_scan_bctc_text.yaml"
)
YAML_TABLE_PATH = os.getenv(
    "P1A_YAML_TABLE",
    r"D:\1.TLAT\3. ChatBot_project\1_Insurance_Strategy\configs\c2_ocr_scan_bctc_table.yaml"
)

# ---- RATIOS YAML (tuỳ chọn) ----
YAML_RATIO_PATH = os.getenv(
    "P1A_YAML_RATIO",
    r"D:\1.TLAT\3. ChatBot_project\1_Insurance_Strategy\configs\c2_ocr_scan_bctc_ratios.yaml"
)

# --- Print config only once ---
_CONFIG_PRINTED = False

def _print_config_once(args):
    """In cấu hình P1A đúng 1 lần."""
    global _CONFIG_PRINTED
    if _CONFIG_PRINTED:
        return
    _CONFIG_PRINTED = True
    print("=== CẤU HÌNH (P1A prelight OCR) ===")
    print(f"📂 PRELIGHT_DIR : {args.prelight_dir}")
    print(f"📦 OUTPUT_DIR   : {args.out}")
    print(f"📝 YAML_TEXT    : {YAML_TEXT_PATH}")
    print(f"📐 YAML_TABLE   : {YAML_TABLE_PATH}")
    print(f"🧮 YAML_RATIOS  : {YAML_RATIO_PATH}")
    print(f"🔤 OCR_LANG     : {args.ocr_lang}")
    print(f"⚙️  OCR_CFG      : {args.ocr_cfg}")
    print(f"🧭 Pages        : {args.start} → {args.end or 'END'}")
    print(f"🎯 prefer       : {args.prefer}")
    print(f"🧹 CLEAN MODE   : {args.clean}  (ask/y/files/a/n)")
    print("=============================")

# --- YAML cache (duy nhất) ---
_yaml_cache = {"text": None, "table": None, "ratio": None}

def _load_yaml_ratio():
    if not os.path.isfile(YAML_RATIO_PATH):
        return {}
    if _yaml_cache.get("ratio") is None:
        with open(YAML_RATIO_PATH, "r", encoding="utf-8") as f:
            _yaml_cache["ratio"] = yaml.safe_load(f) or {}
    return _yaml_cache["ratio"]

def _load_yaml_cfg():
    """Load YAML cấu hình chỉ 1 lần (cache). Trả về (cfg_table, cfg_text)."""
    # Kiểm tra tồn tại để báo lỗi rõ ràng
    if not os.path.isfile(YAML_TABLE_PATH):
        raise FileNotFoundError(f"Không tìm thấy YAML TABLE: {YAML_TABLE_PATH}")
    if not os.path.isfile(YAML_TEXT_PATH):
        raise FileNotFoundError(f"Không tìm thấy YAML TEXT:  {YAML_TEXT_PATH}")

    if _yaml_cache["table"] is None:
        with open(YAML_TABLE_PATH, "r", encoding="utf-8") as f:
            _yaml_cache["table"] = yaml.safe_load(f) or {}
    if _yaml_cache["text"] is None:
        with open(YAML_TEXT_PATH, "r", encoding="utf-8") as f:
            _yaml_cache["text"] = yaml.safe_load(f) or {}

    return _yaml_cache["table"], _yaml_cache["text"]

# [ADD] hỗ trợ clean/append
APPEND_MODE = False  # sẽ bật True trong main() khi --clean a
CLEAN_FILES = False  # ⚑ mới: xoá theo từng file trang trong phạm vi start–end

import src.env  # ✅ đảm bảo nạp .env.active và set OPENAI_API_KEY

# === GPT enhancer (module ngoài) + công tắc ngay trong code ===
from src.c1_gpt_bctc_outputs import enhance_table_with_gpt as gpt_fix_table

USE_GPT = True   # True = BẬT GPT; False = TẮT GPT

# ========= ĐƯỜNG DẪN MẶC ĐỊNH (theo yêu cầu) =========
PRELIGHT_DIR_DEFAULT = r"D:\1.TLAT\3. ChatBot_project\1_Insurance_Strategy\\outputs\outputs0\c0_ocr_scan_bctc_to_png"
OUTPUT_DIR_DEFAULT   = r"D:\1.TLAT\3. ChatBot_project\1_Insurance_Strategy\outputs\outputs1\c1_ocr_scan_bctc_outputs"

# ========= Cấu hình Tesseract =========
OCR_LANG_DEFAULT = "vie+eng"
OCR_CFG_DEFAULT  = "--psm 6"

TESSERACT_CMD = os.environ.get("TESSERACT_CMD", None)
if TESSERACT_CMD and os.path.isfile(TESSERACT_CMD):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# ========= Utils =========
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def clean_txt_chars(s: str) -> str:
    """Loại ký tự rác + normalize space."""
    if not s: return ""
    s = re.sub(r"[|¦•∙·]+", " ", s)
    s = re.sub(r"[^\S\r\n]{2,}", " ", s)  # nhiều space liên tiếp -> 1
    s = re.sub(r"[^\x20-\x7E\u00A0-\uFFFF]", " ", s)  # bỏ ký tự vô hình
    return s.strip()

def _sha1_text(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8")).hexdigest()

# ========= Detect unit / company / period / language =========
_UNIT_SCALES = {
    "đồng": 1, "dong": 1, "vnd": 1, "vnđ": 1, "vn dong": 1, "vn d": 1,
    "nghìn đồng": 1_000, "ngan dong": 1_000, "nghin dong": 1_000, "ngàn đồng": 1_000,
    "triệu đồng": 1_000_000, "trieu dong": 1_000_000,
    "tỷ đồng": 1_000_000_000, "ty dong": 1_000_000_000,
}
def _strip_vn_accents(s: str) -> str:
    rep = {
        "đ":"d","ơ":"o","ô":"o","ư":"u","ă":"a","â":"a","á":"a","à":"a","ả":"a","ã":"a","ạ":"a",
        "é":"e","è":"e","ẻ":"e","ẽ":"e","ẹ":"e","í":"i","ì":"i","ỉ":"i","ĩ":"i","ị":"i",
        "ó":"o","ò":"o","ỏ":"o","õ":"o","ọ":"o","ú":"u","ù":"u","ủ":"u","ũ":"u","ụ":"u",
        "ý":"y","ỳ":"y","ỷ":"y","ỹ":"y","ỵ":"y"
    }
    s = (s or "").lower()
    for k,v in rep.items():
        s = s.replace(k, v)
    return re.sub(r"\s+", " ", s).strip()

def detect_unit_details(text: str):
    """Trả về (unit_raw, unit_normalized, unit_multiplier)."""
    if not text:
        return None, None, None
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    head = lines[:30]
    donvi_like = re.compile(r"d[o0]n\s*v[ij][t]?in[h]?", re.IGNORECASE)
    unit_raw = None
    for ln in head:
        if donvi_like.search(_strip_vn_accents(ln)):
            unit_raw = ln; break
    if not unit_raw:
        for ln in head:
            if re.search(r"\b(VN[ĐD]|VND|đồng|dong|triệu|trieu|tỷ|ty)\b", ln, flags=re.IGNORECASE):
                unit_raw = ln; break
    if not unit_raw:
        return None, None, None
    norm = _strip_vn_accents(unit_raw)
    for key, mul in _UNIT_SCALES.items():
        if key in norm:
            return unit_raw, "VND", mul
    if any(k in norm for k in ["vnd", "vn d", "vnđ", "dong", "d0ng", "d ong"]):
        return unit_raw, "VND", 1
    if re.search(r"\bVND\b", unit_raw, flags=re.IGNORECASE):
        return unit_raw, "VND", 1
    return unit_raw, None, None

def detect_company(text: str) -> Optional[str]:
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in (text or "").splitlines()]
    head = [ln for ln in lines[:20] if ln]
    scored = []
    for ln in head:
        low = ln.lower()
        score = 0
        score += 2 if "công ty" in low or "cong ty" in low else 0
        score += 2 if "bảo hiểm" in low or "bao hiem" in low else 0
        score += 1 if "tổng" in low or "tong" in low else 0
        if score:
            scored.append((score, len(ln), ln))
    if scored:
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return scored[0][2]
    return head[0] if head else None

VN_DATE_1 = r"ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})"
VN_DATE_2 = r"ngay\s+(\d{1,2})\s+thang\s+(\d{1,2})\s+nam\s+(\d{4})"
DATE_DMY  = r"(\d{1,2})[\/\-.](\d{1,2})[\/\-.](\d{4})"
DATE_YMD  = r"(\d{4})[\/\-.](\d{1,2})[\/\-.](\d{1,2})"
MONTHS = {
    "january":1,"february":2,"march":3,"april":4,"may":5,"june":6,
    "july":7,"august":8,"september":9,"october":10,"november":11,"december":12
}
EN_MDY = r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})"
EN_DMY = r"(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December),?\s+(\d{4})"
QUARTER_1 = r"Quý\s*(I{1,3}|IV|\d)\s*[\/\- ]\s*(\d{4})"
QUARTER_2 = r"Quý\s*(I{1,3}|IV|\d)\s*(?:năm|nam)?\s*(\d{4})"
QUARTER_3 = r"Q\s*([1-4])\s*[\/\- ]?\s*(\d{4})"

def _norm_ymd(y: int, m: int, d: int) -> str:
    try:
        return f"{int(y):04d}-{int(m):02d}-{int(d):02d}"
    except Exception:
        return ""

def _month_to_int(name: str) -> Optional[int]:
    return MONTHS.get((name or "").lower().strip())

def _roman_to_int(s: str) -> Optional[int]:
    s = (s or "").upper().strip()
    return {"I":1,"II":2,"III":3,"IV":4}.get(s)

def _quarter_to_date(q: int, y: int) -> str:
    q = int(q)
    return {1:f"{y:04d}-03-31",2:f"{y:04d}-06-30",3:f"{y:04d}-09-30"}.get(q, f"{y:04d}-12-31")

def _extract_iso_from_fragment(frag: str) -> Optional[str]:
    """
    Trả về chuỗi ISO 'YYYY-MM-DD' nếu bắt được 1 ngày trong đoạn frag, ngược lại None.
    Hỗ trợ: ngày VN, DMY, YMD, EN (MDY, DMY).
    """
    s = frag or ""

    m = re.search(VN_DATE_1, s, flags=re.IGNORECASE)
    if m:
        d, mth, y = [int(x) for x in m.groups()]
        return _norm_ymd(y, mth, d)

    m = re.search(VN_DATE_2, s, flags=re.IGNORECASE)
    if m:
        d, mth, y = [int(x) for x in m.groups()]
        return _norm_ymd(y, mth, d)

    m = re.search(DATE_DMY, s)
    if m:
        d, mth, y = [int(x) for x in m.groups()]
        return _norm_ymd(y, mth, d)

    m = re.search(DATE_YMD, s)
    if m:
        y, mth, d = [int(x) for x in m.groups()]
        return _norm_ymd(y, mth, d)

    m = re.search(EN_MDY, s, flags=re.IGNORECASE)
    if m:
        mon, d, y = m.groups()
        mm = _month_to_int(mon)
        if mm:
            return _norm_ymd(int(y), int(mm), int(d))

    m = re.search(EN_DMY, s, flags=re.IGNORECASE)
    if m:
        d, mon, y = m.groups()
        mm = _month_to_int(mon)
        if mm:
            return _norm_ymd(int(y), int(mm), int(d))

    return None


def detect_period_full(text: str) -> Dict[str, Optional[str]]:
    """
    Trả về:
      {
        "period": "YYYY-MM-DD" | None,
        "period_text": "đoạn gốc khớp regex" | None,
        "period_basis": "year_ended" | "as_of" | "quarter_ended" | "period_ended" | None
      }
    Ưu tiên: year_ended > as_of > quarter_ended > bất kỳ ngày bắt được.
    """
    t = text or ""

    # 1) Năm tài chính kết thúc ngày ...
    patt_year_end_vn = r"(?:năm\s+tài\s+chính\s+)?k[êe]t\s*th[úu]c\s*ngày\s+([^\n,]+)"
    patt_year_end_en = r"for\s+the\s+year\s+ended\s+([^\n,]+)"
    for patt in (patt_year_end_vn, patt_year_end_en):
        m = re.search(patt, t, flags=re.IGNORECASE)
        if m:
            iso = _extract_iso_from_fragment(m.group(1))
            if iso:
                return {
                    "period": iso,
                    "period_text": m.group(0).strip(),
                    "period_basis": "year_ended",
                }

    # 2) Tại ngày ...
    patt_as_of_vn = r"(?:tại|tai)\s+ngày\s+([^\n,]+)"
    patt_as_of_en = r"as\s+at\s+([^\n,]+)"
    for patt in (patt_as_of_vn, patt_as_of_en):
        m = re.search(patt, t, flags=re.IGNORECASE)
        if m:
            iso = _extract_iso_from_fragment(m.group(1))
            if iso:
                return {
                    "period": iso,
                    "period_text": m.group(0).strip(),
                    "period_basis": "as_of",
                }

    # 3) Quý ...
    for patt in (QUARTER_1, QUARTER_2, QUARTER_3):
        m = re.search(patt, t, flags=re.IGNORECASE)
        if m:
            q_raw, y = m.groups()
            q = int(q_raw) if q_raw.isdigit() else _roman_to_int(q_raw)
            if q and y:
                iso = _quarter_to_date(q, int(y))
                return {
                    "period": iso,
                    "period_text": m.group(0).strip(),
                    "period_basis": "quarter_ended",
                }

    # 4) Bắt bất kỳ ngày nào còn lại trong text (DMY/YMD/EN)
    iso = _extract_iso_from_fragment(t)
    if iso:
        return {
            "period": iso,
            "period_text": iso,
            "period_basis": "period_ended",
        }

    return {"period": None, "period_text": None, "period_basis": None}


def detect_language(text: str) -> str:
    if not text: return "vi"
    vi_marks = re.findall(r"[ăâêôơưđáàảãạéèẻẽẹíìỉĩịóòỏõọúùủũụýỳỷỹỵ]", text.lower())
    if len(vi_marks) >= 3: return "vi"
    if re.search(r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\b", text, re.IGNORECASE):
        return "en"
    return "vi"

# ========= TSV reflow =========
def _is_amount_text(s: str) -> bool:
    return bool(re.search(r"\d{1,3}(?:[.,]\d{3}){2,}(?:\s|$)", s))

def _is_new_code(s: str) -> bool:
    if re.match(r"^\s*\d{3}(?:\.\d+)?\s*$", s): return True   # 131 / 131.1
    if re.match(r"^\s*(I|II|III|IV|V)\.?\s*$", s, re.I): return True
    return False

def reflow_lines_from_tsv_dict(data: Dict[str, List],
                               y_tol: int = 4) -> str:
    """
    Reflow AN TOÀN:
    - Ghép token trong CÙNG 1 (block,par,line) của Tesseract.
    - Không merge chéo dòng.
    - Sau khi có danh sách dòng, chèn xuống dòng cứng trước các 'mã số' và 'mục lớn'.
    """
    n = len(data.get("text", []))
    groups: Dict[Tuple[int,int,int], List[int]] = {}
    for i in range(n):
        t = (data["text"][i] or "").strip()
        if not t:
            continue
        try:
            conf = float(data["conf"][i])
        except Exception:
            conf = 0.0
        if conf < -1:
            continue
        key = (int(data["block_num"][i]), int(data["par_num"][i]), int(data["line_num"][i]))
        groups.setdefault(key, []).append(i)

    # 1) dựng các dòng theo tesseract
    lines = []
    for key, idxs in sorted(groups.items()):
        idxs = sorted(idxs, key=lambda k: int(data["left"][k]))
        txt = " ".join((data["text"][k] or "").strip() for k in idxs).strip()
        if not txt:
            continue
        y = int(min(int(data["top"][k]) for k in idxs))
        lines.append({"y": y, "text": txt})

    # 2) sắp xếp theo Y
    lines.sort(key=lambda r: r["y"])

    # 3) ép xuống dòng trước mẫu 'mã số' / 'mục lớn'
    #    - 3 chữ số, có thể có .x (vd 131.1)
    #    - các mục I., II., III., IV., V.
    #    - trước số tiền lớn (ít nhất 3 nhóm nghìn)
    code_pat = re.compile(r"(?<!^)\s(?=\d{3}(?:\.\d+)?\b)")
    roman_pat = re.compile(r"(?<!^)\s(?=(?:I|II|III|IV|V)\.?\b)")
    money_pat = re.compile(r"(?<!^)\s(?=\d{1,3}(?:[.,]\d{3}){2,}\b)")

    out_lines: List[str] = []
    for ln in lines:
        s = ln["text"]
        # chèn newline an toàn
        s = code_pat.sub("\n", s)
        s = roman_pat.sub("\n", s)
        s = money_pat.sub("\n", s)
        # chuẩn hoá: tách những đoạn sau khi chèn newline
        out_lines.extend([p.strip() for p in s.split("\n") if p.strip()])

    return "\n".join(out_lines)

# ========= HELPERS: detect/extract table PIPE from text =========
_PIPE_BLOCK_PAT = re.compile(r"<<<PIPE>>>[\s\S]*?<<<END>>>", re.IGNORECASE)

def _extract_pipe_blocks(text: str) -> List[str]:
    """Trả về danh sách block PIPE (ưu tiên khối được code đánh dấu <<<PIPE>>> … <<<END>>>).
       Nếu không có, sẽ tìm các đoạn có dấu '|' lặp nhiều dòng (>=3 dòng có '|')."""
    if not text: 
        return []
    blocks = []
    # 1) Ưu tiên block có đánh dấu
    for m in re.finditer(r"<<<PIPE>>>\s*(.*?)\s*<<<END>>>", text, flags=re.S|re.I):
        blk = m.group(1).strip()
        if blk:
            blocks.append(blk)
    if blocks:
        return blocks

    # 2) Tự tìm đoạn có '|' (ít nhất 3 dòng)
    lines = text.splitlines()
    cur = []
    for ln in lines:
        if "|" in ln:
            cur.append(ln)
        else:
            if len(cur) >= 3:
                blocks.append("\n".join(cur).strip())
            cur = []
    if len(cur) >= 3:
        blocks.append("\n".join(cur).strip())
    return blocks

def _autobuild_pipe_from_text(text: str) -> Optional[str]:
    """
    Quét prose OCR để kéo ra các dòng dạng:
      <code(3 số hoặc 3 số.chỉ mục)> <tên...> <end 1.234.567.890> <begin 1.234.567.890>
    Trả về chuỗi PIPE (CODE|NAME|NOTE|END|BEGIN) hoặc None nếu bắt được quá ít dòng.
    """
    if not text:
        return None
    # số có ít nhất 2 nhóm nghìn (ổn với BCTC)
    money = r"\d{1,3}(?:[.,]\d{3}){2,}"
    # code 3 số (có thể có .x) đứng đầu một cụm
    pat = re.compile(
        rf"(?P<code>\b\d{{3}}(?:\.\d+)?\b)\s+"
        rf"(?P<name>[^\n]*?)\s+"
        rf"(?P<end>{money})\s+"
        rf"(?P<begin>{money})",
        flags=re.IGNORECASE
    )

    rows = []
    for m in pat.finditer(text):
        code  = m.group("code").strip()
        name  = re.sub(r"\s{2,}", " ", m.group("name")).strip(" .:-")
        end   = m.group("end")
        begin = m.group("begin")
        # lọc tên quá ngắn/vô nghĩa
        if len(name) < 3:
            continue
        rows.append(f"{code} | {name} |  | {end} | {begin}")

    if len(rows) < 5:
        return None

    header = "code | name | note | end | begin"
    return header + "\n" + "\n".join(rows)

_NUMERIC_HEAVY = re.compile(r"\d{1,3}(?:[.,]\d{3}){1,}")  # số có nhóm nghìn

def _is_numeric_table_like(pipe_block: str) -> bool:
    """Có nhiều số dạng 1.234.567 hoặc 1,234,567 → coi là bảng số liệu."""
    if not pipe_block or "|" not in pipe_block:
        return False
    num_lines = 0
    for ln in pipe_block.splitlines():
        if "|" not in ln:
            continue
        if len(_NUMERIC_HEAVY.findall(ln)) >= 1:
            num_lines += 1
    return num_lines >= 3  # ít nhất 3 dòng có số

def _replace_block(original_text: str, old_block: str, new_block: str) -> str:
    """
    Nếu old_block nằm trong <<<PIPE>>>…<<<END>>> thì thay tại chỗ.
    Nếu không, chúng ta sẽ **append** một block kiểm tra ở cuối (không phá văn bản).
    """
    m = re.search(r"(<<<PIPE>>>\s*)(.*?)(\s*<<<END>>>)", original_text, flags=re.S|re.I)
    if m and old_block.strip() in m.group(2):
        return original_text[:m.start(2)] + new_block + original_text[m.end(2):]
    return original_text.rstrip() + "\n\n### [TABLE→GPT CHECK]\n<<<PIPE>>>\n" + new_block + "\n<<<END>>>"

def gpt_numbers_only_validate(text_raw: str, image_path: Optional[str], meta_partial: dict) -> str:
    """
    - Tìm các block PIPE trong text_raw
    - Lọc block có nhiều số liệu → chỉ những block này mới cho GPT kiểm tra
    - GPT đối chiếu ẢNH (image_path) và chỉnh lại số (giữ PIPE)
    - Thay thế block trong text (nếu block có đánh dấu) hoặc append kết quả ở cuối
    """
    try:
        if not text_raw:
            return text_raw
        if not image_path or not os.path.exists(image_path):
            return text_raw

        pipe_blocks = _extract_pipe_blocks(text_raw)

        # Nếu chưa có PIPE, thử tự dựng PIPE thô từ prose OCR
        if not pipe_blocks:
            auto_pipe = _autobuild_pipe_from_text(text_raw)
            if auto_pipe:
                text_raw = text_raw.rstrip() + "\n\n<<<PIPE>>>\n" + auto_pipe + "\n<<<END>>>"
                pipe_blocks = [auto_pipe]
            else:
                return text_raw

        out_text = text_raw
        for blk in pipe_blocks:
            if not _is_numeric_table_like(blk):
                continue

            try:
                from PIL import Image as _PILImage
                img = _PILImage.open(image_path).convert("RGB")
            except Exception as e:
                print(f"⚠️ Không mở được ảnh cho GPT validate: {e}")
                return text_raw

            fixed = gpt_fix_table(
                table_text_cleaned=blk,
                image_pil=img,
                meta={
                    "company_hint": meta_partial.get("company"),
                    "period_hint":  meta_partial.get("period"),
                },
                mode="financial",
                model=os.getenv("GPT_OCR_MODEL", "gpt-4o-mini"),
                temperature=0.0,
                log_diag=False,
            )

            if fixed and "|" in fixed and fixed.strip() != blk.strip():
                out_text = _replace_block(out_text, old_block=blk, new_block=fixed)

        return out_text

    except Exception as e:
        print(f"⚠️ gpt_numbers_only_validate error → fallback: {e}")
        return text_raw

def _cleanup_number_str(raw: str, g):
    if raw is None: return None
    s = str(raw)
    # drop chars
    for ch in (g.get("number_cleanup", {}).get("drop_chars") or []):
        s = s.replace(ch, "")
    # regex fixes
    for pat in (g.get("number_cleanup", {}).get("fix_patterns") or []):
        s = re.sub(pat["from"], pat["to"], s)
    s = s.strip()
    return s

def _to_amount_or_none(s: str) -> Optional[int]:
    if not s: return None
    t = s.strip()
    # normalize thousand separator .
    t = t.replace(",", ".")
    # keep only digits, dots and minus
    t = re.sub(r"[^0-9\.\-]", "", t)
    if t in ("", "-", "--"): return None
    # remove dots to make integer (thousands)
    t = t.replace(".", "")
    try:
        return int(t)
    except Exception:
        return None

def _fix_code_ocr_dots(code: str) -> str:
    """
    Sửa lỗi OCR kiểu 1.511 -> 151.1, 1.512 -> 151.2.
    Quy tắc: nếu khớp ^\d\.\d{3}$ thì bỏ dấu chấm đầu, chèn dấu chấm trước chữ số cuối.
    """
    c = (code or "").strip()
    if re.match(r"^\d\.\d{3}$", c):   # ví dụ '1.511'
        raw = c.replace(".", "")      # '1511'
        return raw[:-1] + "." + raw[-1]  # '151.1'
    return c

def _pull_note_from_name(name: str) -> tuple[str, Optional[str]]:
    """
    Nếu 'name' kết thúc bằng token thuyết minh (4, 6, 7, 5.2, 15.1...), tách sang cột note.
    Trả về (name_clean, note_val).
    """
    s = (name or "").strip()
    m = re.search(r"(.*?)(\b\d{1,2}(?:\.\d)?\b)\s*$", s)
    if not m:
        return s, None
    left, token = m.group(1).strip(), m.group(2).strip()
    if len(token) <= 4:  # '4','6','7','5.2','15.1'…
        return left, token
    return s, None

def _auto_layout_for_pipe(pipe_block: str):
    # đếm cột ở dòng dài nhất để suy ước L5/L4/L3
    lines = [ln for ln in pipe_block.splitlines() if "|" in ln]
    if not lines: return "L5"
    cols = max(len([c for c in ln.split("|")]) for ln in lines)
    if cols >= 5: return "L5"
    if cols == 4: return "L4"
    return "L3A"  # tối thiểu 3 cột: name | end | begin

def _parse_pipe_to_rows(pipe_block: str, cfg):
    g = (cfg or {}).get("globals", {})

    # --- FALLBACK LAYOUTS (dùng khi YAML không có globals.layouts) ---
    DEFAULT_LAYOUTS = {
        "L5":  {"column_map": {"code":0, "name":1, "note":2, "end":3, "begin":4}},
        "L4":  {"column_map": {"code":0, "name":1,             "end":2, "begin":3},
                "fill_missing": {"note": None}},
        "L3A": {"column_map": {"name":0, "end":1, "begin":2},
                "fill_missing": {"code":"", "note": None},
                "infer_code_from_name": True},
    }

    layouts = (g.get("layouts") or {})
    lay = _auto_layout_for_pipe(pipe_block)

    # Ưu tiên YAML, nếu thiếu thì dùng DEFAULT_LAYOUTS
    layout = layouts.get(lay) or layouts.get("L5") or DEFAULT_LAYOUTS.get(lay) or DEFAULT_LAYOUTS["L5"]

    col_map = layout.get("column_map", DEFAULT_LAYOUTS["L5"]["column_map"])
    fill_missing = (layout.get("fill_missing") or {})

    rows = []
    for ln in pipe_block.splitlines():
        if "|" not in ln: 
            continue
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        def _get(colname):
            idx = col_map.get(colname)
            if idx is None:
                return fill_missing.get(colname)
            return cells[idx] if idx < len(cells) else fill_missing.get(colname)

        code  = (_get("code") or "").strip()
        code  = _fix_code_ocr_dots(code)              # sửa lỗi OCR kiểu 1.511 -> 151.1

        name  = (_get("name") or "").strip()
        note  = (_get("note") or None)

        # nếu cột note đang trống, thử kéo "thuyết minh" (4, 6, 7, 5.2, 15.1, ...) từ cuối name
        if not (note and str(note).strip()):
            name, pulled_note = _pull_note_from_name(name)
            if pulled_note:
                note = pulled_note

        end   = _cleanup_number_str(_get("end"), g)
        begin = _cleanup_number_str(_get("begin"), g)

        # infer code by name if requested (hiện để trống)
        if not code and layout.get("infer_code_from_name"):
            pass

        rows.append({
            "code": code,
            "name": name,
            "note": note,
            "end":  _to_amount_or_none(end),
            "begin": _to_amount_or_none(begin),
            "_raw_end": end, "_raw_begin": begin,
        })
    return rows

def _index_by_code(rows):
    idx = {}
    for r in rows:
        c = (r["code"] or "").strip()
        if c:
            idx[c] = r
    return idx

def _sum_codes(codes: List[str], idx: Dict[str, dict], col: str) -> Optional[int]:
    total = 0
    any_used = False
    for code in codes:
        code = str(code).strip()
        sign = 1
        if code.startswith("-"):
            sign = -1
            code = code[1:].strip()
        row = idx.get(code)
        if not row: 
            continue
        val = row.get(col)
        if val is None:
            continue
        total += sign * int(val)
        any_used = True
    return total if any_used else None

def _apply_section_rules(rows, cfg_section):
    """Trả về (changed, rows). Áp dụng rules eq vào cả end & begin khi có."""
    changed = False
    idx = _index_by_code(rows)
    rules = cfg_section.get("rules") or []
    for rule in rules:
        # rule: {eq: ["PARENT", ["C1","C2","-C3"]]}
        if "eq" not in rule: 
            continue
        parent, children = rule["eq"][0], rule["eq"][1]
        for col in ("end", "begin"):
            parent_row = idx.get(str(parent))
            if not parent_row: 
                continue
            child_sum = _sum_codes(children, idx, col)
            if child_sum is None: 
                continue
            parent_val = parent_row.get(col)
            if parent_val != child_sum:
                # auto-fix: set parent = sum(children)
                parent_row[col] = child_sum
                changed = True
    return changed, rows

def _apply_cross_formulas(rows, cfg_globals):
    """Áp dụng cross_formulas dạng '270 = 100 + 200' cho end/begin nếu có."""
    changed = False
    idx = _index_by_code(rows)
    for f in (cfg_globals.get("cross_formulas") or []):
        # f: {name: "...", end: "270 = 100 + 200", begin: "270 = 100 + 200"}
        for col in ("end", "begin"):
            expr = f.get(col)
            if not expr: 
                continue
            m = re.match(r"\s*(\S+)\s*=\s*(.+)$", expr)
            if not m: 
                continue
            parent = m.group(1).strip()
            rhs = [x.strip() for x in m.group(2).replace("+"," + ").replace("-"," - ").split() if x.strip() not in {"+","-"}]
            # rebuild with signs (simple parse)
            signed = []
            prev_sign = +1
            for tok in m.group(2).split():
                tok = tok.strip()
                if tok == "+": prev_sign = +1; continue
                if tok == "-": prev_sign = -1; continue
                signed.append(("-" + tok) if prev_sign < 0 else tok)
            s = _sum_codes(signed, idx, col)
            if s is None: 
                continue
            prow = idx.get(parent)
            if prow and prow.get(col) != s:
                prow[col] = s
                changed = True
    return changed, rows

def _rows_to_pipe(rows, cfg):
    g = (cfg or {}).get("globals", {})
    cols = g.get("column_names") or {"code":"code","name":"name","note":"note","end":"end","begin":"begin"}
    def _fmt(v):
        if v is None: return ""
        s = f"{int(v):,}".replace(",", ".")
        return s
    out = []
    header = f"{cols['code']} | {cols['name']} | {cols.get('note','note')} | {cols['end']} | {cols['begin']}"
    out.append(header)
    for r in rows:
        out.append(f"{r.get('code','')} | {r.get('name','')} | {r.get('note','') or ''} | {_fmt(r.get('end'))} | {_fmt(r.get('begin'))}")
    return "\n".join(out)

# ====== NAME-BASED MATCHING HELPERS ======
def _strip_accents(s: str) -> str:
    rep = {
        "đ":"d","Đ":"D","ơ":"o","Ơ":"O","ô":"o","Ô":"O","ư":"u","Ư":"U",
        "ă":"a","Ă":"A","â":"a","Â":"A","á":"a","Á":"A","à":"a","À":"A","ả":"a","Ả":"A","ã":"a","Ã":"A","ạ":"a","Ạ":"A",
        "é":"e","É":"E","è":"e","È":"E","ẻ":"e","Ẻ":"E","ẽ":"e","Ẽ":"E","ẹ":"e","Ẹ":"E",
        "í":"i","Í":"I","ì":"i","Ì":"I","ỉ":"i","Ỉ":"I","ĩ":"i","Ĩ":"I","ị":"i","Ị":"I",
        "ó":"o","Ó":"O","ò":"o","Ò":"O","ỏ":"o","Ỏ":"O","õ":"o","Õ":"O","ọ":"o","Ọ":"O",
        "ú":"u","Ú":"U","ù":"u","Ù":"U","ủ":"u","Ủ":"U","ũ":"u","Ũ":"U","ụ":"u","Ụ":"U",
        "ý":"y","Ý":"Y","ỳ":"y","Ỳ":"Y","ỷ":"y","Ỷ":"Y","ỹ":"y","Ỹ":"Y","ỵ":"y","Ỵ":"Y",
    }
    t = (s or "").strip()
    for k,v in rep.items():
        t = t.replace(k, v)
    t = re.sub(r"\s+", " ", t)
    return t.lower()

def _norm_name(s: str, name_aliases: Dict[str,str]) -> str:
    t = _strip_accents(s)
    # map alias không dấu → tên chuẩn (có dấu) rồi lại chuẩn hoá để khớp ổn định
    if name_aliases:
        for alias, canon in name_aliases.items():
            if _strip_accents(alias) == t:
                return _strip_accents(canon)
    return t

def _index_by_name(rows: list, cfg_table: dict) -> Dict[str, list]:
    """Trả về map {normalized_name: [rows...]} để tra theo tên."""
    aliases = (cfg_table.get("globals") or {}).get("name_aliases") or {}
    idx: Dict[str, list] = {}
    for r in rows:
        nm = _norm_name(r.get("name",""), aliases)
        if nm:
            idx.setdefault(nm, []).append(r)
    return idx

def _find_first_by_name(name: str, name_idx: Dict[str,list], cfg_table: dict):
    nm = _norm_name(name, (cfg_table.get("globals") or {}).get("name_aliases") or {})
    lst = name_idx.get(nm) or []
    return lst[0] if lst else None

def _sum_by_names(names: list, name_idx: Dict[str,list], col: str, cfg_table: dict):
    total = 0
    any_used = False
    for n in names:
        sign = 1
        key = n
        if isinstance(n, str) and n.startswith("-"):
            sign = -1
            key = n[1:]
        row = _find_first_by_name(key, name_idx, cfg_table)
        if not row: 
            continue
        val = row.get(col)
        if val is None:
            continue
        total += sign * int(val)
        any_used = True
    return total if any_used else None

def _norm_noacc(s: str) -> str:
    return _strip_accents(s or "")

def _first_value_by_keysyn(rows: list, name_idx: dict, synonyms: dict, key: str, col: str):
    """Tìm giá trị theo nhóm synonym key (vd 'current_assets' → ['Tài sản ngắn hạn', ...])."""
    alts = (synonyms or {}).get(key) or []
    for nm in alts:
        row = _find_first_by_name(nm, name_idx, {"globals":{"name_aliases":{}}})
        if row:
            v = row.get(col)
            if v is not None:
                return v
    return None

def _compute_ratios_from_pipe(
    pipe_block: str,
    ratio_cfg: dict,
    table_cfg: dict | None = None
) -> list[dict]:
    """
    Tính các ratio từ khối PIPE.
    - ratio_cfg: YAML ratios (synonyms + ratios)
    - table_cfg: YAML TABLE (để parse layout/alias). Có thể None; khi đó _parse_pipe_to_rows dùng fallback layout.
    Trả về: [{"name": str, "end": float|None, "begin": float|None}, ...]
    """
    if not ratio_cfg:
        return []

    # 1) Parse PIPE → rows theo layout (ưu tiên table_cfg nếu có)
    rows = _parse_pipe_to_rows(pipe_block, cfg=(table_cfg or {}))

    # 2) Lập index theo code và theo tên (dùng name_aliases nếu có trong table_cfg)
    name_aliases = ((table_cfg or {}).get("globals") or {}).get("name_aliases") or {}
    name_idx = _index_by_name(rows, {"globals": {"name_aliases": name_aliases}})
    code_idx = _index_by_code(rows)

    # 3) Synonyms & định nghĩa ratios
    syn = ratio_cfg.get("synonyms") or {}
    defs = ratio_cfg.get("ratios") or []

    def _first_value_by_keysyn(key: str, col: str):
        """
        Lấy giá trị theo:
          - Nếu key giống mã số (vd '270' hoặc '151.1') → tra code_idx
          - Ngược lại coi như 'nhóm semantic' (vd 'current_assets') → duyệt synonyms[key] theo tên
        """
        # a) key là mã số?
        if re.match(r"^\d{2,3}(?:\.\d+)?$", str(key).strip()):
            row = code_idx.get(str(key).strip())
            return row.get(col) if row else None

        # b) key là nhãn semantic → thử từng synonym (tên chỉ tiêu)
        for cand in syn.get(key, []):
            nm_norm = _norm_name(cand, name_aliases)
            lst = name_idx.get(nm_norm) or []
            if lst:
                v = lst[0].get(col)
                if v is not None:
                    return v
        return None

    out: list[dict] = []
    for rdef in defs:
        name = rdef.get("name")
        num_keys = rdef.get("numerator") or []
        den_keys = rdef.get("denominator") or []
        if not name or not num_keys or not den_keys:
            continue

        row_out = {"name": name, "end": None, "begin": None}

        for col in ("end", "begin"):
            # numerator: lấy giá trị đầu tiên tìm thấy trong danh sách keys
            num = None
            for k in num_keys:
                num = _first_value_by_keysyn(k, col)
                if num is not None:
                    break

            # denominator
            den = None
            for k in den_keys:
                den = _first_value_by_keysyn(k, col)
                if den is not None:
                    break

            if num is not None and den not in (None, 0):
                row_out[col] = float(num) / float(den)

        if row_out["end"] is not None or row_out["begin"] is not None:
            out.append(row_out)

    return out

def _format_ratios_as_text(items: list[dict]) -> str:
    if not items: return ""
    lines = ["\n### [RATIOS]"]
    for it in items:
        e = f"{it['end']:.4f}" if it.get("end") is not None else "—"
        b = f"{it['begin']:.4f}" if it.get("begin") is not None else "—"
        lines.append(f"- {it['name']}: END={e} | BEGIN={b}")
    return "\n".join(lines) + "\n"

def yaml_validate_and_autofix_pipe(pipe_block: str, cfg: dict) -> str:
    """Parse → apply rules (code-based) → cross formulas → name-based rules → render lại PIPE."""
    # 0) Parse & chuẩn bị
    rows = _parse_pipe_to_rows(pipe_block, cfg)
    g = (cfg or {}).get("globals", {}) or {}

    changed = False

    # 1) RULES THEO MÃ (sections.assets / sections.equity_liab)
    bs = (cfg or {}).get("balance_sheet") or {}
    if bs:
        for sec_name in ("assets", "equity_liab"):
            sec = (bs.get("sections") or {}).get(sec_name)
            if sec:
                c, rows = _apply_section_rules(rows, sec)
                changed = changed or c

    # 2) CROSS FORMULAS THEO MÃ (globals.cross_formulas: "270 = 100 + 200")
    c, rows = _apply_cross_formulas(rows, g)
    changed = changed or c

    # 3) RULES THEO TÊN (globals.parent_children_by_name, balance_sheet.cross_sheet_rules_by_name)
    try:
        parent_children_by_name = g.get("parent_children_by_name") or {}
        name_idx = _index_by_name(rows, cfg)

        # 3a) Cha–con theo TÊN
        for parent_name, children_names in parent_children_by_name.items():
            prow = _find_first_by_name(parent_name, name_idx, cfg)
            if not prow:
                continue
            for col in ("end", "begin"):
                s = _sum_by_names(children_names, name_idx, col, cfg)
                if s is None:
                    continue
                if prow.get(col) != s:
                    prow[col] = s
                    changed = True

        # 3b) Cross-sheet by name (ví dụ: "Tổng tài sản" == "Tổng nguồn vốn")
        for rule in (bs.get("cross_sheet_rules_by_name") or []):
            eq = rule.get("equals") or {}
            left_name  = eq.get("left")
            right_name = eq.get("right")
            if not left_name or not right_name:
                continue
            lrow = _find_first_by_name(left_name, name_idx, cfg)
            rrow = _find_first_by_name(right_name, name_idx, cfg)
            if not lrow or not rrow:
                continue
            for col in ("end", "begin"):
                lv = lrow.get(col); rv = rrow.get(col)
                if lv is None and rv is not None:
                    lrow[col] = rv; changed = True
                elif rv is None and lv is not None:
                    rrow[col] = lv; changed = True
                elif lv is not None and rv is not None and lv != rv:
                    # Ưu tiên lấy theo bên phải (có thể đổi tuỳ ý)
                    lrow[col] = rv
                    changed = True
    except Exception as _e:
        print("⚠️ NAME-based validator warning:", _e)

    # 4) Xuất lại PIPE (nếu không đổi, trả về như cũ)
    return _rows_to_pipe(rows, cfg)

# ========= OCR 1 ảnh =========
def ocr_image_to_text_and_meta(img_bgr, ocr_lang: str, ocr_cfg: str) -> Tuple[str, str]:
    try:
        txt_raw = pytesseract.image_to_string(img_bgr, lang=ocr_lang, config=(ocr_cfg or "").strip())
    except pytesseract.TesseractNotFoundError:
        print("⚠️ Không tìm thấy Tesseract. Set TESSERACT_CMD tới tesseract.exe.")
        txt_raw = ""
    except Exception as e:
        print(f"⚠️ Lỗi Tesseract (string): {e}")
        txt_raw = ""

    try:
        cfg2 = (ocr_cfg or "").strip()
        cfg2 = re.sub(r"--psm\s+\d+", "", cfg2).strip()
        cfg2 = (cfg2 + " --psm 4 preserve_interword_spaces=1").strip()
        tsv = pytesseract.image_to_data(img_bgr, lang=ocr_lang, config=cfg2, output_type=TessOutput.DICT)
        txt = reflow_lines_from_tsv_dict(tsv) or txt_raw
        # xoá các khoảng trắng kép và gạch nối dài
        txt = re.sub(r"[ \t]{2,}", " ", txt)
        txt = re.sub(r"([\-]{3,})", "-", txt)

    except Exception as e:
        print(f"⚠️ Lỗi Tesseract (data): {e} → dùng string")
        txt = txt_raw

    txt = clean_txt_chars(txt)

    # Meta con
    unit_raw, unit_norm, unit_mul = detect_unit_details(txt)
    company  = detect_company(txt)
    period_d = detect_period_full(txt)
    title, stmt = detect_report_title_and_statement(txt)
    lang = detect_language(txt)
    return txt, json.dumps({
        "unit": unit_norm, "unit_raw": unit_raw, "unit_multiplier": unit_mul,
        "company": company,
        "period": period_d.get("period"),
        "period_text": period_d.get("period_text"),
        "period_basis": period_d.get("period_basis"),
        "report_title": title,
        "statement": stmt,
        "language": lang or "vi"
    })

def _norm_letters(s: str) -> str:
    s = _strip_vn_accents(s or "")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def detect_report_title_and_statement(text: str):
    if not text: return None, None
    norm_all = _norm_letters(text)
    if re.search(r"b[a]?ng\s+can\s+do[i1l]\s+ke\s+toan", norm_all): return "Bảng cân đối kế toán", "balance_sheet"
    if "balance sheet" in norm_all or "statement of financial position" in norm_all: return "Bảng cân đối kế toán", "balance_sheet"
    if "bao cao ket qua hoat dong kinh doanh" in norm_all or "income statement" in norm_all:
        return "Báo cáo kết quả hoạt động kinh doanh", "income_statement"
    if "bao cao luu chuyen tien te" in norm_all or "cash flow" in norm_all:
        return "Báo cáo lưu chuyển tiền tệ", "cash_flow"
    if "bao cao thay doi von chu so huu" in norm_all or "changes in equity" in norm_all:
        return "Báo cáo thay đổi vốn chủ sở hữu", "equity_changes"
    if all(k in norm_all for k in ["ma so", "thuyet minh", "so cuoi nam", "so dau nam"]):
        return "Bảng cân đối kế toán", "balance_sheet"
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        longest = sorted(lines[:15], key=len, reverse=True)[0]
        return longest, None
    return None, None

# ========= Tìm & xử lý ảnh prelight =========
def find_prelight_pages(prelight_root: str) -> Dict[Tuple[str,int], Dict[str,str]]:
    """
    Trả về map {(base,page): {"bin": path_bin_or_None, "orig": path_orig_or_None}}
    Dò theo pattern: <base>_pageNN_bin.png / <base>_pageNN_orig.png (đệ quy)
    """
    pages: Dict[Tuple[str,int], Dict[str,str]] = {}
    for path in glob.glob(os.path.join(prelight_root, "**", "*_page??_bin.png"), recursive=True):
        name = os.path.basename(path)
        m = re.match(r"(.+?)_page(\d+)_bin\.png$", name, re.IGNORECASE)
        if not m: continue
        base, pg = m.group(1), int(m.group(2))
        pages.setdefault((base, pg), {})["bin"] = path
    for path in glob.glob(os.path.join(prelight_root, "**", "*_page??_orig.png"), recursive=True):
        name = os.path.basename(path)
        m = re.match(r"(.+?)_page(\d+)_orig\.png$", name, re.IGNORECASE)
        if not m: continue
        base, pg = m.group(1), int(m.group(2))
        pages.setdefault((base, pg), {})["orig"] = path
    return pages

def apply_yaml_text_rules(text: str, ytext: dict) -> str:
    """Làm sạch prose theo YAML TEXT: number_cleanup, alias tên/mã, v.v."""
    if not text or not ytext:
        return text

    g = (ytext.get("globals") or {})

    # number_cleanup
    ncl = (g.get("number_cleanup") or {})
    for ch in (ncl.get("drop_chars") or []):
        text = text.replace(ch, "")
    for it in (ncl.get("fix_patterns") or []):
        try:
            text = re.sub(it.get("from", ""), it.get("to", ""), text)
        except re.error:
            pass
    if ncl.get("thousand_grouping"):
        # đổi dấu phẩy nghìn -> chấm
        text = re.sub(r"(?<=\d),(?=\d{3}\b)", ".", text)

    # alias mã (sửa lỗi OCR mã)
    for raw, ali in (g.get("code_aliases") or {}).items():
        text = re.sub(rf"(?<!\d){re.escape(raw)}(?!\d)", ali, text)

    # alias tên (phẳng)
    for raw, ali in (g.get("name_aliases_flat") or {}).items():
        text = re.sub(re.escape(raw), ali, text, flags=re.IGNORECASE)

    return text

def process_one_page(out_root: str, base: str, page_no: int,
                     src_img_path: str, ocr_lang: str, ocr_cfg: str,
                     source_pdf: Optional[str] = None) -> None:
    out_dir = os.path.join(out_root)  # mirror: prelight đã mirror; p1a chỉ đặt chung 1 root theo yêu cầu
    ensure_dir(out_dir)

    # --- OCR ---
    bgr = cv2.cvtColor(np.array(Image.open(src_img_path).convert("RGB")), cv2.COLOR_RGB2BGR)
    txt, meta_partial_json = ocr_image_to_text_and_meta(bgr, ocr_lang, ocr_cfg)
    meta_partial = json.loads(meta_partial_json)

    # --- GPT numbers-only (đối chiếu PNG, chỉ sửa bảng số) ---
    if USE_GPT:
        print("🧠 GPT table-check: ON (numbers-only)")
        txt = gpt_numbers_only_validate(txt, src_img_path, meta_partial)
    else:
        print("🧠 GPT table-check: OFF")

    # === YAML VALIDATE/AUTOFIX (chỉ chạy trên các bảng PIPE) ===
    cfg_table, cfg_text = _load_yaml_cfg()          # nạp YAML table/text
    txt = apply_yaml_text_rules(txt, cfg_text)
    
    pipe_blocks = _extract_pipe_blocks(txt) or []    # tách block PIPE từ txt

    any_changed = False
    for blk in pipe_blocks:
        if not _is_numeric_table_like(blk):
            continue
        try:
            # áp quy tắc theo MÃ (eq/cross) + theo TÊN (parent_children_by_name/cross_sheet_by_name)
            fixed_pipe = yaml_validate_and_autofix_pipe(blk, cfg_table)
        except Exception as e:
            print("⚠️ YAML validator error:", e)
            continue

        if fixed_pipe and fixed_pipe.strip() != blk.strip():
            # _replace_block(original_text, old_block, new_block) — gọi theo thứ tự đối số
            txt = _replace_block(txt, blk, fixed_pipe)
            any_changed = True

    print("🔧 YAML validator:", "adjusted tables" if any_changed else "no changes")

    # === RATIOS (tùy chọn) ===
    # Lấy block PIPE số liệu đầu tiên sau khi đã GPT + YAML fix
    first_numeric = None
    for blk in _extract_pipe_blocks(txt):
        if _is_numeric_table_like(blk):
            first_numeric = blk
            break

    ratios = []
    if first_numeric:
        ratio_cfg = _load_yaml_ratio()
        ratios = _compute_ratios_from_pipe(first_numeric, ratio_cfg, table_cfg=cfg_table)
    else:
        ratios = []

    # ---- Ghi file (append-only nếu APPEND_MODE=True) ----
    text_path = os.path.join(out_dir, f"{base}_page{page_no}_text.txt")
    if APPEND_MODE and os.path.exists(text_path):
        print(f"↩️ Bỏ qua (đã có): {os.path.basename(text_path)}")
    else:
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(txt)

    meta = {
        "ratios": ratios,
        "file": base,
        "page": page_no,
        "company": meta_partial.get("company"),
        "unit": meta_partial.get("unit"),
        "unit_raw": meta_partial.get("unit_raw"),
        "unit_multiplier": meta_partial.get("unit_multiplier"),
        "currency_code": "VND",
        "period": meta_partial.get("period"),
        "period_text": meta_partial.get("period_text"),
        "period_basis": meta_partial.get("period_basis"),
        "report_title": meta_partial.get("report_title"),
        "statement": meta_partial.get("statement"),
        "language": meta_partial.get("language") or "vi",
        "source_image": os.path.abspath(src_img_path),
        "source_pdf": source_pdf,
        "preprocess": True,       # ảnh đã qua prelight
        "ocr_lang": ocr_lang,
        "ocr_cfg": ocr_cfg,
        "text_sha1": _sha1_text(txt),
    }
    meta_path = os.path.join(out_dir, f"{base}_page{page_no}_meta.json")
    if APPEND_MODE and os.path.exists(meta_path):
        print(f"↩️ Bỏ qua (đã có): {os.path.basename(meta_path)}")
    else:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"📝 Saved: {os.path.basename(text_path)}, {os.path.basename(meta_path)}")

def _unlink_quiet(path: str) -> None:
    """Xoá file nếu tồn tại, không ném exception khi không có."""
    try:
        if path and os.path.exists(path):
            os.remove(path)
            print(f"🗑️  Xoá: {os.path.basename(path)}")
    except Exception as e:
        print(f"⚠️ Không xoá được {path}: {e}")

def run_ocr_on_prelight(prelight_dir: str, out_dir: str,
                        start_page: int, end_page: Optional[int],
                        ocr_lang: str, ocr_cfg: str,
                        prefer: str = "bin") -> None:
    """
    prefer: "bin" | "orig" | "auto" (auto = dùng bin nếu có, không có thì orig)
    """
    mapping = find_prelight_pages(prelight_dir)
    if not mapping:
        print(f"⚠️ Không tìm thấy ảnh *_page??_(bin|orig).png trong: {prelight_dir}")
        return

    # Lọc theo khoảng trang
    keys = sorted(mapping.keys(), key=lambda x: x[1])
    keys = [k for k in keys if (start_page <= k[1] <= (end_page or 10**9))]
    if not keys:
        print("⚠️ Không có trang phù hợp với --start/--end.")
        return

    print(f"📂 Input (prelight): {prelight_dir}")
    print(f"📦 Output         : {out_dir}")
    print(f"🧭 Trang          : {keys[0][1]} → {keys[-1][1]} (lọc theo tham số)")
    ensure_dir(out_dir)

    # [NEW] nếu chọn clean=files thì xoá đúng các file trang sắp chạy
    if CLEAN_FILES:
        for (base, pg) in keys:
            text_path = os.path.join(out_dir, f"{base}_page{pg}_text.txt")
            meta_path = os.path.join(out_dir, f"{base}_page{pg}_meta.json")
            _unlink_quiet(text_path)
            _unlink_quiet(meta_path)

    for (base, pg) in keys:
        cand = mapping[(base, pg)]
        img_path = None
        if prefer == "bin":
            img_path = cand.get("bin") or cand.get("orig")
        elif prefer == "orig":
            img_path = cand.get("orig") or cand.get("bin")
        else:  # auto
            img_path = cand.get("bin") or cand.get("orig")
        if not img_path:
            print(f"⚠️ Thiếu ảnh cho {base}_page{pg:02d}"); continue
        print(f"🔎 {base}_page{pg:02d} → {os.path.basename(img_path)}")
        process_one_page(out_dir, base, pg, img_path, ocr_lang, ocr_cfg, source_pdf=None)

# ========= CLI =========
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("P1A — OCR từ ảnh prelight (_bin/_orig) → TXT + META")
    p.add_argument("--prelight-dir", type=str, default=PRELIGHT_DIR_DEFAULT,
                   help="Thư mục chứa ảnh prelight (mirror, có *_pageNN_bin.png / *_pageNN_orig.png)")
    p.add_argument("--out", type=str, default=OUTPUT_DIR_DEFAULT,
                   help="Thư mục output (đặt phẳng theo yêu cầu)")
    p.add_argument("--start", type=int, default=1, help="Trang bắt đầu")
    p.add_argument("--end", type=int, default=None, help="Trang kết thúc (inclusive)")
    p.add_argument("--prefer", choices=["bin","orig","auto"], default="bin",
                   help="Ưu tiên dùng ảnh nào (mặc định: bin)")
    p.add_argument("--ocr-lang", type=str, default=OCR_LANG_DEFAULT, help="Ngôn ngữ OCR (mặc định: vie+eng)")
    p.add_argument("--ocr-cfg",  type=str, default=OCR_CFG_DEFAULT,  help="Tesseract config (mặc định: --psm 6)")
    p.add_argument("--clean", choices=["ask","y","a","n","files"], default="ask",
                   help="ask: hỏi; y: xoá cả thư mục; files: xoá từng file trang trong phạm vi; a: append-only; n: bỏ qua nếu đã tồn tại")
    return p

def main():
    global APPEND_MODE, CLEAN_FILES  # phải đứng TRƯỚC mọi phép gán trong hàm

    args = build_argparser().parse_args()

    # In cấu hình 1 lần
    _print_config_once(args)

    # Chuẩn bị thư mục output theo --clean
    out_dir = args.out
    if os.path.exists(out_dir):
        choice = args.clean
        if choice == "ask":
            choice = input(f"⚠️ Output '{out_dir}' đã tồn tại. y=xoá, files=xoá từng file, a=append, n=bỏ qua: ").strip().lower()
        if choice == "y":
            shutil.rmtree(out_dir, ignore_errors=True); print(f"🗑️ Đã xoá {out_dir}")
        elif choice == "files":
            # không xoá thư mục; đánh dấu để run_ocr_on_prelight tự xoá từng file cần chạy
            pass
        elif choice == "n":
            print("⏭️ Bỏ qua P1A."); return
        elif choice == "a":
            print(f"➕ Giữ {out_dir}, chỉ ghi file mới.")
        else:
            print("❌ Lựa chọn không hợp lệ → bỏ qua."); return

    os.makedirs(out_dir, exist_ok=True)

    # bật cờ theo chế độ clean
    APPEND_MODE = (args.clean == "a")
    CLEAN_FILES = (args.clean == "files")
    print(f"🧠 USE_GPT (code switch) = {USE_GPT}")

    # Chạy OCR trên ảnh prelight
    run_ocr_on_prelight(
        prelight_dir=args.prelight_dir,
        out_dir=args.out,
        start_page=args.start,
        end_page=args.end,
        ocr_lang=args.ocr_lang,
        ocr_cfg=args.ocr_cfg,
        prefer=args.prefer
    )
    print("\n✅ Hoàn tất P1A. Kiểm tra *_text.txt và *_meta.json tại thư mục output.")

if __name__ == "__main__":
    main()
