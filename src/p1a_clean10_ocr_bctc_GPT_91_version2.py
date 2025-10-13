# -*- coding: utf-8 -*-
"""
P1A (GPT) — Hybrid Auto-Route (OpenCV+TSV → Paddle) [FULL/Hardened]
- Auto score chất lượng bảng và route engine:
    1) TSV/KMeans (nhẹ, nhanh)  [DEFAULT]
    2) Nếu score xấu → Paddle PP-Structure
    3) So sánh score, chọn bản tốt hơn (best-of-two)
- Lọc header/caption “rác bảng” trước YAML/validator (pattern mạnh + contains)
- Gom dòng theo Y-tol thích ứng (theo median height)
- Neo hai cột số (END/BEGIN) theo tiêu đề ở phần đầu trang; fallback KMeans + histogram 2 số
- Chuẩn hoá số kiểu Việt + tách “2 số dính liền”
- Lọc token rác 1–2 ký tự, gộp mảnh NAME; kiểm tra hàng hợp lệ
- Validator gắt và Narrator chống NaN
- Cờ: --narrator y|n (mặc định y), --dpi (mặc định 360), default engines = TSV/Tesseract

I/O:
- Mỗi file input → 1 TXT (### [PAGE XX] [TEXT]/[TABLE]/[TABLE→ROW-NARR]) + 1 META.json
- --split-debug sinh thêm *_TEXT.txt / *_TABLE.txt
"""

from __future__ import annotations
import os, re, io, sys, json, glob, argparse, hashlib
from typing import Optional, Dict, List, Tuple, Any

import numpy as np
import cv2
from PIL import Image
import pytesseract
from pytesseract import Output as TessOutput

import pdfplumber
import fitz  # PyMuPDF
try:
    from docx import Document
    HAS_DOCX = True
except Exception:
    HAS_DOCX = False
from tqdm import tqdm
import yaml
import pandas as pd
from sklearn.cluster import KMeans
# ---- RAW toggle (bỏ qua YAML & prefilter) ----
P1A_RAW_MODE = False   # True = chạy thô, False = chạy theo YAML

# --- PaddleOCR (optional) ---
try:
    from paddleocr import PaddleOCR, PPStructure
    _HAS_PADDLE = True
except Exception:
    _HAS_PADDLE = False

# cache singleton cho Paddle (tránh khởi tạo lại)
_PADDLE_OCR = None
_PPSTRUCT = None
def get_paddle_ocr(lang="vi", use_gpu=False):
    global _PADDLE_OCR
    if _PADDLE_OCR is None:
        _PADDLE_OCR = PaddleOCR(lang=lang, use_angle_cls=True, use_gpu=use_gpu, show_log=False)
    return _PADDLE_OCR
def get_ppstructure(lang="en", use_gpu=False):
    global _PPSTRUCT
    if _PPSTRUCT is None:
        _PPSTRUCT = PPStructure(show_log=False, use_gpu=use_gpu, lang=lang, layout=False, table=True)
    return _PPSTRUCT

# --- GPT enhancer (optional) ---
try:
    import src.env  # nạp OPENAI_API_KEY từ .env.active nếu có
except Exception:
    pass
try:
    from src.p1a_gpt_ocr_bctc import enhance_table_with_gpt
    _HAS_GPT_ENHANCER = True
except Exception:
    _HAS_GPT_ENHANCER = False

# ====== ĐƯỜNG DẪN MẶC ĐỊNH ======
INPUT_ROOT_DEFAULT  = r"D:\1.TLAT\3. ChatBot_project\1_Insurance_Strategy\inputs\c_financial_reports_test"
OUTPUT_ROOT_DEFAULT = r"D:\1.TLAT\3. ChatBot_project\1_Insurance_Strategy\outputs\p1a_clean10_ocr_bctc_GPT_91_version2"
YAML_TABLE_DEFAULT  = r"D:\1.TLAT\3. ChatBot_project\1_Insurance_Strategy\configs\p1a_clean10_ocr_bctc_table.yaml"
YAML_TEXT_DEFAULT   = r"D:\1.TLAT\3. ChatBot_project\1_Insurance_Strategy\configs\p1a_clean10_ocr_bctc_text.yaml"

# ====== OCR CONFIG ======
OCR_LANG_DEFAULT = "vie+eng"
OCR_CFG_SENTENCE = "--psm 4 -c preserve_interword_spaces=1"
OCR_CFG_TSV = "--psm 6 -c preserve_interword_spaces=1"

# ====== REGEX & HELPERS ======
CODE_LINE = re.compile(r"(?m)^(?:\s*\|?\s*)\d{3}(?:\.\d+)?\b")   # 131, 151.1...
MONEY     = re.compile(r"\b\d{1,3}(?:[.,]\d{3}){2,}\b")          # 1.234.567
NOTE_TAG  = re.compile(r"(?<!^)\s(?=\d+\([a-z]\))", re.I)
CODE_3    = re.compile(r"^\d{3}(?:\.\d+)?$")                     # mã hợp lệ

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def clean_txt_chars(s: str) -> str:
    import re as _re
    if not s: return ""
    s = _re.sub(r"[|¦•∙·]+", " ", s)
    s = _re.sub(r"[^\S\r\n]{2,}", " ", s)
    s = _re.sub(r"[^\x20-\x7E\u00A0-\uFFFF]", " ", s)
    return s.strip()

def _sha1(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8")).hexdigest()

# ====== Nhận diện đơn vị/kỳ ======
_UNIT_SCALES = {
    "đồng": 1, "dong": 1, "vnd": 1, "vnđ": 1, "vn dong": 1, "vn d": 1,
    "nghìn đồng": 1_000, "ngan dong": 1_000, "nghin dong": 1_000, "ngàn đồng": 1_000,
    "triệu đồng": 1_000_000, "trieu dong": 1_000_000,
    "tỷ đồng": 1_000_000_000, "ty dong": 1_000_000_000,
}
def _strip_vn_accents(s: str) -> str:
    rep = {"đ":"d","ơ":"o","ô":"o","ư":"u","ă":"a","â":"a","á":"a","à":"a","ả":"a","ã":"a","ạ":"a",
           "é":"e","è":"e","ẻ":"e","ẽ":"e","ẹ":"e","í":"i","ì":"i","ỉ":"i","ĩ":"i","ị":"i",
           "ó":"o","ò":"o","ỏ":"o","õ":"o","ọ":"o","ú":"u","ù":"u","ủ":"u","ũ":"u","ụ":"u",
           "ý":"y","ỳ":"y","ỷ":"y","ỹ":"y","ỵ":"y"}
    s = (s or "").lower()
    for k,v in rep.items(): s = s.replace(k, v)
    return re.sub(r"\s+", " ", s).strip()

def detect_unit(text: str):
    t = (text or "")
    norm = _strip_vn_accents(t)
    m = re.search(r"don\s*vi\s*tinh\s*[:：]\s*([a-zA-ZđĐ]{2,10})", norm, re.I)
    unit_raw = None
    if m: unit_raw = m.group(1).upper().strip()
    else:
        m2 = re.search(r"\b(vnd|vnđ)\b", norm, re.I)
        if m2: unit_raw = "VND"
    unit = None; mult = None
    if unit_raw:
        ur = unit_raw.lower()
        if "vnd" in ur or "vnđ" in ur: unit = "VND"; mult = 1
        else:
            for k, v in _UNIT_SCALES.items():
                if k.replace(" ", "") in ur.replace(" ", ""):
                    unit = unit_raw; mult = v; break
    return unit, mult, unit_raw

VN_DATE_1 = r"ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})"
VN_DATE_2 = r"ngay\s+(\d{1,2})\s+thang\s+(\d{1,2})\s+nam\s+(\d{4})"
DATE_DMY  = r"(\d{1,2})[\/\-.](\d{1,2})[\/\-.](\d{4})"
DATE_YMD  = r"(\d{4})[\/\-.](\d{1,2})[\/\-.](\d{1,2})"
EN_MDY    = r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})"
EN_DMY    = r"(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December),?\s+(\d{4})"
MONTHS = {m.lower():i+1 for i,m in enumerate([
    "January","February","March","April","May","June","July","August","September","October","November","December"]) }
def _norm_ymd(y,m,d):
    try: return f"{int(y):04d}-{int(m):02d}-{int(d):02d}"
    except: return ""
def _month_to_int(name: str): return MONTHS.get((name or "").lower().strip())
def _extract_iso_from_fragment(frag: str):
    s = frag or ""
    m = re.search(VN_DATE_1, s, re.I) or re.search(VN_DATE_2, s, re.I)
    if m: d,mn,y = [int(x) for x in m.groups()]; return _norm_ymd(y,mn,d)
    m = re.search(DATE_DMY, s)
    if m: d,mn,y = [int(x) for x in m.groups()]; return _norm_ymd(y,mn,d)
    m = re.search(DATE_YMD, s)
    if m: y,mn,d = [int(x) for x in m.groups()]; return _norm_ymd(y,mn,d)
    m = re.search(EN_MDY, s, re.I)
    if m:
        mon, d, y = m.groups(); mm = _month_to_int(mon)
        if mm: return _norm_ymd(int(y), int(mm), int(d))
    m = re.search(EN_DMY, s, re.I)
    if m:
        d, mon, y = m.groups(); mm = _month_to_int(mon)
        if mm: return _norm_ymd(int(y), int(mm), int(d))
    return None
def detect_period_full(text: str):
    t = text or ""
    patt_year_end_vn = r"(?:năm\s+tài\s+chính\s+)?k[êe]t\s*th[úu]c\s*ngày\s+([^\n,]+)"
    patt_year_end_en = r"for\s+the\s+year\s+ended\s+([^\n,]+)"
    for patt in (patt_year_end_vn, patt_year_end_en):
        m = re.search(patt, t, re.I)
        if m:
            iso = _extract_iso_from_fragment(m.group(1))
            if iso: return {"period": iso, "period_text": m.group(0).strip(), "period_basis": "year_ended"}
    patt_as_of_vn = r"(?:tại|tai)\s+ngày\s+([^\n,]+)"
    patt_as_of_en = r"as\s+at\s+([^\n,]+)"
    for patt in (patt_as_of_vn, patt_as_of_en):
        m = re.search(patt, t, re.I)
        if m:
            iso = _extract_iso_from_fragment(m.group(1))
            if iso: return {"period": iso, "period_text": m.group(0).strip(), "period_basis": "as_of"}
    iso = _extract_iso_from_fragment(t)
    if iso: return {"period": iso, "period_text": iso, "period_basis": "period_ended"}
    return {"period": None, "period_text": None, "period_basis": None}

# ====== TSV REFLOW (TEXT) ======
def reflow_lines_from_tsv_dict(data: Dict[str, List]) -> str:
    import re as _re
    n = len(data.get("text", []))
    groups: Dict[Tuple[int,int,int], List[int]] = {}
    for i in range(n):
        t = (data["text"][i] or "").strip()
        if not t: continue
        try: conf = float(data["conf"][i])
        except: conf = 0.0
        if conf < -1: continue
        key = (int(data["block_num"][i]), int(data["par_num"][i]), int(data["line_num"][i]))
        groups.setdefault(key, []).append(i)
    lines = []
    for key, idxs in sorted(groups.items()):
        idxs = sorted(idxs, key=lambda k: int(data["left"][k]))
        txt = " ".join((data["text"][k] or "").strip() for k in idxs).strip()
        if not txt: continue
        y = int(min(int(data["top"][k]) for k in idxs))
        lines.append({"y": y, "text": txt})
    lines.sort(key=lambda r: r["y"])
    code_pat  = _re.compile(r"(?<!^)\s(?=\d{3}(?:\.\d+)?\b)")
    roman_pat = _re.compile(r"(?<!^)\s(?=(?:I|II|III|IV|V)\.?\b)")
    money_pat = _re.compile(r"(?<!^)\s(?=\d{1,3}(?:[.,]\d{3}){2,}\b)")
    out_lines = []
    for ln in lines:
        s = ln["text"]
        s = code_pat.sub("\n", s)
        s = roman_pat.sub("\n", s)
        s = money_pat.sub("\n", s)
        s = NOTE_TAG.sub("\n", s)
        out_lines.extend([p.strip() for p in s.split("\n") if p.strip()])
    return "\n".join(out_lines)

# ====== YAML CLEAN (TABLE/TEXT) ======
def apply_yaml_clean(text: str, yaml_rules: dict, mode: str) -> str:
    import re as _re
    cfg = yaml_rules or {}
    gl  = (cfg.get("globals") or {})
    ncl = (gl.get("number_cleanup") or {}) if mode == "table" else (gl.get("text_cleanup") or {})
    for ch in (ncl.get("drop_chars") or []):
        text = text.replace(ch, "")
    for pat in (ncl.get("fix_patterns") or []):
        fr, to = pat.get("from",""), pat.get("to","")
        try: text = _re.sub(fr, to, text)
        except _re.error: pass
    if ncl.get("thousand_grouping", False) and mode == "table":
        text = _re.sub(r"(?<=\d)[,](?=\d{3}\b)", ".", text)
        text = _re.sub(r"(\d)\.(\d{1,2})\.(\d{3})(?=[^\d]|$)", r"\1\2.\3", text)
    if mode == "table":
        aliases = gl.get("code_aliases", {}) or {}
        for raw, ali in aliases.items():
            text = _re.sub(rf"(?<!\d){_re.escape(raw)}(?![\d])", ali, text)
    return clean_txt_chars(text)

# ====== OCR CORE ======
def ocr_image_text(img: Image.Image, lang=OCR_LANG_DEFAULT):
    bgr = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    thr  = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)[1]
    bgr  = cv2.cvtColor(thr, cv2.COLOR_GRAY2BGR)
    try:
        raw = pytesseract.image_to_string(bgr, lang=lang, config=OCR_CFG_SENTENCE)
    except Exception as e:
        print(f"⚠️ Tesseract string error: {e}"); raw = ""
    try:
        tsv = pytesseract.image_to_data(bgr, lang=lang, config=OCR_CFG_TSV, output_type=TessOutput.DICT)
    except Exception as e:
        print(f"⚠️ Tesseract data error: {e}"); tsv = None
    try:
        if tsv is not None:
            txt = reflow_lines_from_tsv_dict(tsv) or raw
        else:
            txt = raw
        txt = re.sub(r"[ \t]{2,}", " ", txt)
        txt = re.sub(r"([\-]{3,})", "-", txt)
    except Exception as e:
        print(f"⚠️ Post-TSV reflow/cleanup error: {e} → fallback raw"); txt = raw
    txt = clean_txt_chars(txt)
    unit, mult, unit_raw = detect_unit(raw or txt)
    per = detect_period_full(txt)
    meta = {
        "unit": unit, "unit_raw": unit_raw, "unit_multiplier": mult,
        "company": detect_company(txt),
        "period": per.get("period"), "period_text": per.get("period_text"), "period_basis": per.get("period_basis"),
        "language": detect_language(txt),
        "engine": "tesseract",
    }
    return txt, json.dumps(meta, ensure_ascii=False)

def ocr_image_text_paddle(img: Image.Image, lang="vi", use_gpu=False):
    if not _HAS_PADDLE:
        raise RuntimeError("PaddleOCR chưa sẵn sàng (pip install paddleocr).")
    bgr = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    ocr = get_paddle_ocr(lang=lang, use_gpu=use_gpu)
    result = ocr.ocr(bgr, cls=True)
    lines = []
    for page in result:
        for item in (page or []):
            try:
                box, (text, conf) = item
            except Exception:
                if isinstance(item, list) and len(item) >= 2:
                    box, (text, conf) = item[0], item[1]
                else:
                    continue
            if not (text and str(text).strip()): continue
            ys = [pt[1] for pt in box]; xs = [pt[0] for pt in box]
            cy = sum(ys)/4.0; cx = sum(xs)/4.0
            lines.append((cy, cx, str(text).strip()))
    lines.sort(key=lambda t: (t[0], t[1]))
    txt = "\n".join([t[2] for t in lines])
    txt = clean_txt_chars(txt)
    unit, mult, unit_raw = detect_unit(txt)
    per = detect_period_full(txt)
    meta = {
        "unit": unit, "unit_raw": unit_raw, "unit_multiplier": mult,
        "company": detect_company(txt),
        "period": per.get("period"), "period_text": per.get("period_text"), "period_basis": per.get("period_basis"),
        "language": detect_language(txt),
        "engine": "paddle",
    }
    return txt, json.dumps(meta, ensure_ascii=False)

# ====== TEXT/TABLE detection ======
def detect_language(text: str) -> str:
    if not text: return "vi"
    vi_marks = re.findall(r"[ăâêôơưđáàảãạéèẻẽẹíìỉĩịóòỏõọúùủũụýỳỷỹỵ]", text.lower())
    if len(vi_marks) >= 3: return "vi"
    if re.search(r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\b", text, re.I):
        return "en"
    return "vi"

def detect_company(text: str) -> Optional[str]:
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in (text or "").splitlines()]
    head = [ln for ln in lines[:12] if ln]
    candidates = []
    for ln in head:
        if len(ln) > 120: continue
        low = ln.lower(); score = 0
        if "công ty" in low or "cong ty" in low: score += 2
        if "bảo hiểm" in low or "bao hiem" in low: score += 2
        if "tổng" in low or "tong" in low: score += 1
        upper_ratio = sum(1 for c in ln if c.isupper()) / max(1, len(ln))
        score += 1 if upper_ratio > 0.25 else 0
        if score:
            trimmed = ln.split(".")[0].strip()
            candidates.append((score, len(trimmed), trimmed))
    if candidates:
        candidates.sort(key=lambda x: (x[0], -x[1]), reverse=True)
        return candidates[0][2]
    return head[0] if (head and len(head[0]) <= 120) else None

# ---------- RENDERERS ----------
def render_pdf_page_to_image(pdf_path: str, page_number: int, dpi: int = 380) -> Image.Image:
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_number)
    zoom = dpi / 72.0
    mat  = fitz.Matrix(zoom, zoom)
    pix  = page.get_pixmap(matrix=mat, alpha=False)
    img  = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img

def render_docx_page_to_image(docx_path: str, page_number: int, dpi: int = 360):
    if not HAS_DOCX:
        raise RuntimeError("python-docx chưa sẵn sàng")
    doc = Document(docx_path); text = "\n".join([p.text for p in doc.paragraphs]) or ""
    from PIL import ImageDraw
    W = 2000; pad = 60
    lines = text.splitlines() or [text]
    H = pad*2 + 28*max(20, len(lines))
    img = Image.new("RGB", (W, H), (255,255,255))
    d = ImageDraw.Draw(img); y = pad
    for ln in lines:
        d.text((pad,y), ln[:2500], fill=(0,0,0)); y += 28
    return img

def render_image_file(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")

def iter_pages(path: str, dpi: int):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        with pdfplumber.open(path) as pdf:
            n = len(pdf.pages)
        for i in range(n):
            yield i, render_pdf_page_to_image(path, i, dpi=dpi)
    elif ext == ".docx":
        yield 0, render_docx_page_to_image(path, 0, dpi=dpi)
    elif ext in (".png",".jpg",".jpeg",".tif",".tiff",".bmp"):
        yield 0, render_image_file(path)
    else:
        return

def _looks_like_table(text: str, yaml_table: dict) -> bool:
    if not text:
        return False
    # Loại trừ rõ phần ý kiến kiểm toán / văn bản thuần
    if re.search(r"Ý kiến của Ki(ê|e)m toán|Ki(ê|e)m toán viên|Auditor'?s opinion", text, re.I):
        return False

    det = (yaml_table.get("globals") or {}).get("detection_tokens") or {}
    must_any = det.get("must_have_any") or []

    hit_tokens = any(re.search(re.escape(tok), text, re.I) for tok in must_any)

    code_hits   = len(CODE_LINE.findall(text))
    money_lines = sum(1 for ln in text.splitlines() if len(MONEY.findall(ln)) >= 2)

    # Siết ngưỡng: yêu cầu dày đặc hơn để coi là bảng toàn trang
    dense_struct = (code_hits >= 4 and money_lines >= 4)

    return bool(hit_tokens or dense_struct)


def is_table_page(txt: str, yaml_table: dict) -> bool:
    return _looks_like_table(txt, yaml_table)

# ====== TSV→Hàng/Cột (no-line) ======
def _norm_word(w: str) -> str:
    return re.sub(r"\s+", " ", (w or "").strip())

def _is_numberish(s: str) -> bool:
    return bool(re.search(r"[\d()\-\.,]", s or ""))

def _fix_vn_number(s: str) -> str:
    if not s: return s
    s = s.strip()
    s = re.sub(r"[^\d(),\-\.]", "", s)
    s = s.replace(",", "")
    if s.startswith("(") and s.endswith(")"): s = "-" + s[1:-1]
    return s

def _tsv_dict_to_df(tsv: Dict[str, List]) -> pd.DataFrame:
    n = len(tsv.get("text", []))
    if n == 0: return pd.DataFrame(columns=["left","top","width","height","text","conf"])
    df = pd.DataFrame({k: tsv.get(k, [None]*n) for k in tsv.keys()})
    df = df[(df.conf != -1) & df.text.notna()]
    for col in ("left","top","width","height"):
        try: df[col] = df[col].astype(int)
        except Exception: pass
    df["cx"] = df["left"] + df["width"]/2.0
    df["cy"] = df["top"]  + df["height"]/2.0
    return df

def _cluster_rows(tsv_df: pd.DataFrame, y_tol: int):
    med_h = float(tsv_df["height"].median() or 0)
    y_tol_eff = max(int(y_tol), int(0.35 * med_h) if med_h > 0 else y_tol)
    tsv_df = tsv_df.sort_values(["cy","cx"]).reset_index(drop=True)
    rows, cur, cur_y = [], [], None
    for _, r in tsv_df.iterrows():
        if not _norm_word(r["text"]):
            continue
        if cur_y is None or abs(r["cy"] - cur_y) <= y_tol_eff:
            cur.append(r)
            if cur_y is None: cur_y = r["cy"]
        else:
            rows.append(pd.DataFrame(cur))
            cur = [r]; cur_y = r["cy"]
    if cur: rows.append(pd.DataFrame(cur))
    return rows

_HEADER_HINTS_END   = re.compile(r"(s[ốo]\s*c[uú]ối\s*n[ăa]m|ending\s*balance|current\s*year)", re.I)
_HEADER_HINTS_BEGIN = re.compile(r"(s[ốo]\s*đ[ầa]u\s*n[ăa]m|beginning\s*balance|prior\s*year)", re.I)
# Header kiểu ngày/VND (UIC…): "31/12/2024  VND" | "31/12/2023  VND"
_HEADER_HINTS_DATE  = re.compile(r"(?:\b31|0?[1-9]|[12]\d|3[01])\s*[\/\-.]\s*(?:0?[1-9]|1[0-2])\s*[\/\-.]\s*20\d{2}", re.I)
_HEADER_HINTS_VND   = re.compile(r"\bVND\b|\bVNĐ\b", re.I)

def _anchor_numeric_splits(tsv_df: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
    if tsv_df.empty: return None, None
    H = tsv_df["top"].max() + tsv_df["height"].max()
    cut = 0.35 * H
    head = tsv_df[tsv_df["top"] < cut].copy()
    if head.empty: return None, None

    end_x = []; begin_x = []
    for _, r in head.iterrows():
        txt = _norm_word(str(r["text"]))

        # Case 1: cụm "Số cuối năm / Số đầu năm"
        if _HEADER_HINTS_END.search(txt):   end_x.append(r["cx"])
        if _HEADER_HINTS_BEGIN.search(txt): begin_x.append(r["cx"])

        # Case 2: header kiểu "31/12/2024  VND | 31/12/2023  VND"
        if not end_x or not begin_x:
            if _HEADER_HINTS_DATE.search(txt) and _HEADER_HINTS_VND.search(txt):
                # heuristics: cột bên trái (năm hiện tại) = END, cột bên phải = BEGIN
                # dùng cx để chia 2 cụm theo median
                (end_x if r["cx"] < head["cx"].median() else begin_x).append(r["cx"])

    # nếu sau tất cả vẫn không tìm được, bỏ neo
    if not end_x or not begin_x:
        return None, None

    
    sx_end   = float(np.median(end_x))
    sx_begin = float(np.median(begin_x))
    split1 = max(180, min(sx_end, sx_begin) - 30)
    split2 = max(split1 + 120, (sx_end + sx_begin) / 2.0)
    return float(split1), float(split2)

_SHORT_NOISE = re.compile(r"^(?:[A-ZĐ]{1,2}|[IVX]{1,3}\.?)$")
def _clean_name_token(tok: str) -> str:
    s = (tok or "").strip()
    if CODE_3.match(s): return s
    if _SHORT_NOISE.match(s): return ""
    return s

_AMOUNT_GROUP = re.compile(r"\d{1,3}(?:\.\d{3})+")
def _fallback_two_num_by_hist(row_df: pd.DataFrame, page_w: int) -> Tuple[str,str]:
    cand = row_df.copy()
    cand = cand[cand["text"].astype(str).str.contains(r"\d")]
    cand = cand[cand["cx"] > 0.55*page_w]
    if cand.empty: return "", ""
    texts = " ".join(str(t) for t in cand.sort_values("cx")["text"].tolist())
    hits = _AMOUNT_GROUP.findall(texts)
    if len(hits) >= 2:
        return hits[-2], hits[-1]
    return "", ""

def _infer_columns(row_df: pd.DataFrame, max_cols: int = 5):
    X = row_df[["cx"]].to_numpy()
    if len(X) < 3: return None
    k = min(max_cols, max(2, len(X)))
    km = KMeans(n_clusters=k, n_init="auto", random_state=0).fit(X)
    row_df = row_df.copy(); row_df["col_id"] = km.labels_
    centers = row_df.groupby("col_id")["cx"].mean().sort_values().index.tolist()
    remap = {cid:i for i,cid in enumerate(centers)}
    row_df["col_id"] = row_df["col_id"].map(remap)
    return row_df

def build_generic_table_from_tsv(pil: Image.Image, y_tol: int, ocr_lang: str, max_cols: int = 8) -> str:
    """
    Dựng bảng theo số cột thực tế từ TSV, KHÔNG ép 5 cột.
    Xuất dạng pipe: COL1 | COL2 | ... | COLN (N có thể 3..max_cols).
    """
    bgr = cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    tsv = pytesseract.image_to_data(bgr, lang=ocr_lang, config=OCR_CFG_TSV, output_type=TessOutput.DICT)
    df = _tsv_dict_to_df(tsv)
    if df.empty:
        return ""

    X = df[["cx"]].to_numpy()
    if len(X) < 3:
        return ""

    # chọn k theo elbow đơn giản
    ks, inertias = [], []
    kmax = min(max_cols, max(2, len(X)))
    for k in range(2, kmax + 1):
        km = KMeans(n_clusters=k, n_init="auto", random_state=0).fit(X)
        ks.append(k); inertias.append(float(km.inertia_))
    best_k = ks[0]
    if len(ks) >= 2:
        gains = [inertias[i-1] - inertias[i] for i in range(1, len(inertias))]
        best_k = ks[gains.index(max(gains)) + 1]

    km = KMeans(n_clusters=best_k, n_init="auto", random_state=0).fit(X)
    df2 = df.copy()
    df2["col_id"] = km.labels_
    centers = df2.groupby("col_id")["cx"].mean().sort_values()
    order = {cid:i for i,cid in enumerate(centers.index.tolist())}
    df2["col_id"] = df2["col_id"].map(order)

    header = " | ".join([f"COL{i+1}" for i in range(best_k)])
    lines = [header]
    for row_df in _cluster_rows(df2, y_tol=y_tol):
        row_df = row_df[row_df["text"].astype(str).str.strip().astype(bool)]
        if row_df.empty: 
            continue
        buckets = {i: [] for i in range(best_k)}
        for _, t in row_df.sort_values(["cx"]).iterrows():
            s = _norm_word(str(t["text"]))
            if s: buckets[int(t["col_id"])].append(s)
        row_vals = [" ".join(buckets[i]).strip() for i in range(best_k)]
        lines.append(" | ".join(row_vals))
    return "\n".join(lines)


def _row_is_valid(code, name, end, begin):
    if CODE_3.match((code or "").strip()): return True
    if (end and begin): return True
    return len((name or "").strip()) >= 7


def _extract_code_name_note(left_txt: str) -> Tuple[str, str, str]:
    """
    Tách CODE | NAME | NOTE từ phần trái (không cột số).
    NOTE nhận dạng dạng '5', '5.2', '10(a)' ở cuối tên.
    """
    s = (left_txt or "").strip()
    code = ""; name = s; note = ""
    m = re.match(r"^\s*(\d{3}(?:\.\d+)?)\b\s*(.*)$", s)
    if m:
        code = m.group(1).strip()
        name = (m.group(2) or "").strip()
    mn = re.search(r"(?:^|\s)(\d+(?:\.\d+)?(?:\([a-z]\))?)\s*$", name, flags=re.I)
    if mn and (len(name) - mn.start()) <= 8:
        note = mn.group(1).strip()
        name = name[:mn.start()].strip()
    return code, name, note

def _assign_cols_dynamic(row_df, split1, split2, page_w) -> List[str]:
    """
    Map động token theo vị trí: [CODE | NAME | NOTE | END | BEGIN]
    - Không ép 5 cột cứng; nếu không có NOTE thì để rỗng.
    - Nếu chỉ có 1 cột số, gán vào END, BEGIN để trống.
    """
    left_tokens, mid_tokens, end_tokens, begin_tokens = [], [], [], []
    for _, t in row_df.sort_values("cx").iterrows():
        cx = float(t["cx"])
        s  = _norm_word(str(t["text"]))
        s  = _clean_name_token(s)
        if not s:
            continue
        if cx < (split1 - 16):
            left_tokens.append(s)
        elif cx < (split2 - 16):
            if _is_numberish(s): end_tokens.append(s)
            else:                mid_tokens.append(s)
        elif cx > (split2 + 16):
            if _is_numberish(s): begin_tokens.append(s)
            else:                mid_tokens.append(s)
        else:
            if abs(cx - split1) < abs(cx - split2):
                end_tokens.append(s)
            else:
                begin_tokens.append(s)

    left_txt = " ".join([t for t in left_tokens + mid_tokens if t]).strip()
    code, name, note = _extract_code_name_note(left_txt)

    end_val   = _fix_vn_number(" ".join(end_tokens))
    begin_val = _fix_vn_number(" ".join(begin_tokens))

    if not end_val and not begin_val:
        e2, b2 = _fallback_two_num_by_hist(row_df, page_w=page_w)
        end_val   = _fix_vn_number(e2)
        begin_val = _fix_vn_number(b2)

    # Nếu chỉ trích được 1 cột số → coi là 4 cột (END có số, BEGIN trống)
    if end_val and not begin_val:
        pass
    elif begin_val and not end_val:
        end_val, begin_val = begin_val, ""

    return [code, name, note, end_val, begin_val]



def assemble_financial_rows_from_pil(pil: Image.Image, y_tol: int, lang: str = OCR_LANG_DEFAULT):
    bgr = cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    tsv = pytesseract.image_to_data(bgr, lang=lang, config=OCR_CFG_TSV, output_type=TessOutput.DICT)
    tsv_df = _tsv_dict_to_df(tsv)
    if tsv_df.empty:
        return []

    split1, split2 = _anchor_numeric_splits(tsv_df)
    W = int(tsv_df["left"].max() + tsv_df["width"].max())

    table = []
    for row_df in _cluster_rows(tsv_df, y_tol=y_tol):
        row_df = row_df[row_df["text"].astype(str).str.strip().astype(bool)]
        if row_df.empty: continue

        cols = [""]*5

        if split1 is not None and split2 is not None:
            code, name, note, end_val, begin_val = _assign_cols_dynamic(row_df, split1, split2, page_w=W)
            if not _row_is_valid(code, name, end_val, begin_val):
                continue

            # >>> ADD HERE
            if os.environ.get("P1A_ROWDEBUG") == "1":
                print(f"[row] code={code!r} | name={name[:60]!r} | note={note!r} | end={end_val!r} | begin={begin_val!r}")

            table.append([code, name, note, end_val, begin_val])
            continue

        # --- fallback KMeans (linh hoạt, không ép 5 cột) ---
        row_df2 = _infer_columns(row_df, max_cols=5)
        if row_df2 is None:
            raw_txt = " ".join(row_df.sort_values("cx")["text"].astype(str))
            code, name, note = _extract_code_name_note(raw_txt)
            e2, b2 = _fallback_two_num_by_hist(row_df, page_w=W)
            end, begin = _fix_vn_number(e2), _fix_vn_number(b2)
            if not end and begin:
                end, begin = begin, ""
            if not _row_is_valid(code, name, end, begin):
                continue       
            if os.environ.get("P1A_ROWDEBUG") == "1":
                print(f"[row] code={code!r} | name={name[:60]!r} | note={note!r} | end={end!r} | begin={begin!r}")
                   
            table.append([code, name, note, end, begin])
            continue

        buckets = {cid: [] for cid in sorted(row_df2["col_id"].unique())}
        for _, t in row_df2.iterrows():
            txt = _clean_name_token(_norm_word(str(t["text"])))
            if not txt:
                continue
            buckets[int(t["col_id"])].append(txt)

        centers = row_df2.groupby("col_id")["cx"].mean().sort_values().index.tolist()

        code, name, note, end, begin = "", "", "", "", ""
        numeric_cols = []
        for cid in centers:
            cell = " ".join(buckets.get(cid, [])).strip()
            if _is_numberish(cell):
                numeric_cols.append(cid)

        if len(numeric_cols) >= 2:
            numeric_cols_sorted = sorted(
                numeric_cols,
                key=lambda c: row_df2[row_df2["col_id"]==c]["cx"].mean()
            )
            end_cid, begin_cid = numeric_cols_sorted[-2], numeric_cols_sorted[-1]
            end   = _fix_vn_number(" ".join(buckets.get(end_cid, [])))
            begin = _fix_vn_number(" ".join(buckets.get(begin_cid, [])))
            left_cids = [c for c in centers if c not in (end_cid, begin_cid)]
            left_txt  = " ".join([" ".join(buckets.get(c, [])) for c in left_cids]).strip()
            code, name, note = _extract_code_name_note(left_txt)
        else:
            if numeric_cols:
                end_cid = numeric_cols[-1]
                end = _fix_vn_number(" ".join(buckets.get(end_cid, [])))
                left_cids = [c for c in centers if c != end_cid]
            else:
                end = ""
                left_cids = centers
            left_txt  = " ".join([" ".join(buckets.get(c, [])) for c in left_cids]).strip()
            code, name, note = _extract_code_name_note(left_txt)

        if not _row_is_valid(code, name, end, begin):
            continue
        table.append([code, name, note, end, begin])


    # lọc header lần cuối
    clean = []
    for cols in table:
        if not any(cols): continue
        line = " ".join(cols)
        if re.search(r"VND|Mã\s*số|Thuyết\s*minh|31/|12/", line, re.I): continue
        clean.append(cols)
    return clean

def table_rows_to_pipe(rows: List[List[str]]) -> str:
    lines = ["CODE | NAME | NOTE | END | BEGIN"]
    for r in rows:
        code,name,note,end,begin = (r + ["","","","",""])[:5]
        lines.append(f"{code} | {name} | {note} | {end} | {begin}")
    return "\n".join(lines)

def rows_to_pipe_min(rows: List[Dict[str, str]]) -> str:
    lines = ["CODE | NAME | NOTE | END | BEGIN"]
    for r in rows:
        lines.append(
            f"{r.get('ma','')} | {r.get('chi','')} | {r.get('tm','')} | {r.get('end','')} | {r.get('start','')}"
        )
    return "\n".join(lines)

def rows_to_json_min(rows: List[Dict[str, str]]) -> str:
    return json.dumps(rows, ensure_ascii=False, indent=2)

# === ASCII renderer (top-level, dùng chung cho MIXED/TABLE) ===
def rows_to_ascii(rows: List[Dict[str, str]]) -> str:
    headers = ["Mã số","Chỉ tiêu","Thuyết minh","Số cuối năm","Số đầu năm"]
    if not rows:
        w = {"ma": len(headers[0]), "chi": len(headers[1]), "tm": len(headers[2]),
             "end": len(headers[3]), "start": len(headers[4])}
    else:
        w = {
            "ma":   max(max(len(r.get("ma",""))   for r in rows), len(headers[0])),
            "chi":  max(max(len(r.get("chi",""))  for r in rows), len(headers[1])),
            "tm":   max(max(len(r.get("tm",""))   for r in rows), len(headers[2])),
            "end":  max(max(len(r.get("end",""))  for r in rows), len(headers[3])),
            "start":max(max(len(r.get("start",""))for r in rows), len(headers[4])),
        }
    pad_l = lambda s, ww: (s or "").ljust(ww)
    pad_r = lambda s, ww: (s or "").rjust(ww)
    line = f"+-{'-'*w['ma']}-+-{'-'*w['chi']}-+-{'-'*w['tm']}-+-{'-'*w['end']}-+-{'-'*w['start']}-+"
    out = [line,
           "| "+pad_l(headers[0],w['ma'])+" | "+pad_l(headers[1],w['chi'])+" | "+pad_l(headers[2],w['tm'])+
           " | "+pad_r(headers[3],w['end'])+" | "+pad_r(headers[4],w['start'])+" |",
           line]
    for r in rows:
        out.append("| "+pad_l(r.get('ma',''),w['ma'])+" | "+pad_l(r.get('chi',''),w['chi'])+" | "+
                   pad_l(r.get('tm',''),w['tm'])+" | "+pad_r(r.get('end',''),w['end'])+" | "+
                   pad_r(r.get('start',''),w['start'])+" |")
    out.append(line)
    return "\n".join(out)

# ==== TABLE ROI DETECTOR (morphology) ====
def _find_table_rois(bgr: np.ndarray) -> List[Tuple[int,int,int,int]]:
    """Trả về list bbox (x1,y1,x2,y2) các khung bảng lớn trong trang."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    h, w = thr.shape[:2]
    kx = max(10, w // 250)
    ky = max(10, h // 250)
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (kx*5, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, ky*5))

    horiz = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel_h, iterations=1)
    vert  = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel_v, iterations=1)
    grid  = cv2.add(horiz, vert)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx*3, ky*3))
    grid = cv2.dilate(grid, kernel, iterations=1)

    cnts,_ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rois = []
    area_min = (w*h) * 0.06
    for c in cnts:
        x,y,ww,hh = cv2.boundingRect(c)
        area = ww*hh
        if area < area_min:
            continue
        if hh < h*0.08 and ww > w*0.6:
            continue
        rois.append((x, y, x+ww, y+hh))
    rois.sort(key=lambda b: b[1])
    return rois

def _mask_out_rois(pil: Image.Image, rois: List[Tuple[int,int,int,int]]) -> Image.Image:
    bgr = cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    for (x1,y1,x2,y2) in rois:
        cv2.rectangle(bgr, (x1,y1), (x2,y2), (255,255,255), thickness=-1)
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

# >>> PATCH: preprocess & OCR số cho Paddle
def _preprocess_for_paddle(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    pad = 6
    bgr2 = cv2.copyMakeBorder(bgr, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255,255,255))
    gray = cv2.cvtColor(bgr2, cv2.COLOR_BGR2GRAY)
    thr  = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)[1]
    return cv2.cvtColor(thr, cv2.COLOR_GRAY2BGR)

def normalize_vn_amount(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    neg = s.startswith("(") and s.endswith(")")
    s = re.sub(r"[^\d\.,\s]", "", s)
    s = s.replace(",", ".")
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"\.{2,}", ".", s).strip(".")
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+", s):
        out = s
    else:
        digits = re.sub(r"[^\d]", "", s)
        if len(digits) < 4:
            return ""
        parts = []
        while digits:
            parts.append(digits[-3:])
            digits = digits[:-3]
        out = ".".join(reversed(parts))
    return ("-" + out) if neg else out

def _ocr_crop_number(bgr: np.ndarray, box_xyxy: Tuple[int,int,int,int], lang: str = OCR_LANG_DEFAULT) -> str:
    x1, y1, x2, y2 = [max(0,int(v)) for v in box_xyxy]
    crop = bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return ""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    thr  = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)[1]
    cfg  = "--psm 7 -c tessedit_char_whitelist=0123456789().,- -c classify_bln_numeric_mode=1"
    try:
        s = pytesseract.image_to_string(thr, lang=lang, config=cfg) or ""
    except Exception:
        s = ""
    return normalize_vn_amount(s)

def paddle_table_to_pipe(img: Image.Image, lang="vi", use_gpu=False):
    if not _HAS_PADDLE:
        return None

    bgr_raw = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    bgr = _preprocess_for_paddle(bgr_raw)

    layout_lang = lang if lang in ("en", "ch") else "en"
    table_engine = get_ppstructure(lang=layout_lang, use_gpu=use_gpu)
    result = table_engine(bgr)

    tables = []
    for item in (result or []):
        if item.get("type") != "table":
            continue
        html = (item.get("res") or {}).get("html")
        if not html:
            continue
        try:
            for df in pd.read_html(html):
                if isinstance(df, pd.DataFrame) and not df.empty:
                    tables.append(df)
        except Exception:
            continue

    if not tables:
        return None

    def _flatten_hdr(cols):
        out = []
        for c in cols:
            if isinstance(c, (list, tuple)):
                out.append(" ".join(str(x) for x in c if str(x).strip()))
            else:
                out.append(str(c))
        return [re.sub(r"\s+", " ", (x or "").strip()) for x in out]

    def df_to_pipe_any(df: pd.DataFrame) -> str:
        if df is None or df.empty:
            return ""
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = _flatten_hdr(df.columns)
        df.columns = [re.sub(r"\s+", " ", str(c or "").strip()) for c in df.columns]
        df = df.fillna("")
        lines = [" | ".join([str(c) for c in df.columns.tolist()])]
        for _, r in df.iterrows():
            vals = [str(x) if x is not None else "" for x in r.tolist()]
            lines.append(" | ".join(vals))
        return "\n".join(lines)

    parts = [df_to_pipe_any(df) for df in tables if df is not None]
    return "\n\n".join([p for p in parts if p.strip()])


# ====== PATCH 1 — Lọc header/caption mạnh ======
_DROP_NAME_CONTAINS_DEFAULT = [
    "mã số", "ma so", "thuyết minh", "thuyet minh",
    "31/12/20", "as at", "as of", "for the year ended",
    "vnd", "v n d", "đơn vị", "don vi", "ngày", "ngay",
    "bảng cân đối kế toán", "báo cáo tài chính", "mẫu số", "ban hành theo"
]
_HEADER_STRONG_PATTERNS = [
    r"\b(b[ảa]ng)\s+c[âa]n\s+đ[oó]i\s+k[ếe]\s+to[áa]n\b",
    r"\bm[âa]u\s+s[ốo]\b",
    r"\bban\s+h[àa]nh\s+theo\b",
    r"\b[đd][ơo]n\s+v[ịi]\s*t[íi]nh\b",
    r"\b(as\s+at|as\s+of|for\s+the\s+year\s+ended)\b",
    r"\b(a\.)\s*n[ợo]\s+ph[ảa]i\s+tr[ảa]\b",
    r"\b(b\.)\s*v[ốo]n\s+ch[ủu]\s*s[ởo]\s*h[ữu]\b",
    r"^\s*(\(?\d{3}\)?\s*=|\(?\d{3}\)?\s*[-=+])",
    r"\b(i{1,4}|v|vi|iv)\.?\b",
    r"\bwl\b"
]
_DATE_LINE = re.compile(r"\b(0?[1-9]|[12]\d|3[01])\s*[\/\-.]\s*(0?[1-9]|1[0-2])\s*[\/\-.]\s*(20\d{2})\b")
_VND_LINE  = re.compile(r"\b(vnd|vnđ|ng[hg]ìn\s+đ[ồo]ng|tri[ệe]u\s+đ[ồo]ng|t[ỷy]\s+đ[ồo]ng)\b", re.I)

def _prefilter_table_lines(pipe_text: str, yaml_table: dict) -> str:
    if not pipe_text:
        return pipe_text
    lines = [ln for ln in pipe_text.splitlines() if ln.strip()]
    conf_drop = (yaml_table.get("globals", {}).get("drop_if_name_contains") or [])
    drop_tokens = set([t.lower() for t in conf_drop]) | set(_DROP_NAME_CONTAINS_DEFAULT)
    out = []
    for ln in lines:
        parts = [p.strip() for p in ln.split("|")]
        if len(parts) < 5:
            low = _strip_vn_accents(ln)
            if any(re.search(pat, low, re.I) for pat in _HEADER_STRONG_PATTERNS):
                continue
            if _DATE_LINE.search(low) or _VND_LINE.search(low):
                continue
            out.append(ln); continue
        _, name, *_ = parts
        low = _strip_vn_accents(name)
        if any(re.search(pat, low, re.I) for pat in _HEADER_STRONG_PATTERNS):
            continue
        if _DATE_LINE.search(low) or _VND_LINE.search(low):
            continue

        # chỉ drop nếu KHÔNG có số ở cột END/BEGIN và KHÔNG có mã hợp lệ
        has_amount = False
        try:
            end_txt   = parts[-2].strip() if len(parts) >= 4 else ""
            begin_txt = parts[-1].strip() if len(parts) >= 5 else ""
            has_amount = _is_amount(end_txt) or _is_amount(begin_txt)
        except Exception:
            has_amount = False

        code_ok = False
        try:
            code_ok = CODE_3.match((parts[0] or "").strip()) is not None
        except Exception:
            code_ok = False

        if (not has_amount) and (not code_ok) and any(tok in low for tok in drop_tokens):
            continue

        out.append(ln)
    return "\n".join(out)

# ====== Coerce TEXT → pipe (fallback tuyến tính) ======
def coerce_pipe_table(raw_text: str) -> str:
    if not raw_text: return ""
    out, buf = [], []
    for ln in raw_text.splitlines():
        s = ln.strip()
        if not s: continue
        if re.match(r"^\d{3}(?:\.\d+)?\b", s) and buf:
            out.append(" ".join(buf)); buf = [s]
        else:
            buf.append(s)
    if buf: out.append(" ".join(buf))
    rows = []
    for rec in out:
        monies = MONEY.findall(rec)
        if len(monies) >= 2:
            end, begin = monies[-2], monies[-1]
        else:
            continue
        m = re.search(r"^\s*(\d{3}(?:\.\d+)?)\b", rec)
        if not m: continue
        code = m.group(1)
        tail = rec.rsplit(begin, 1)[0].rsplit(end, 1)[0].strip()
        name_note = tail; note = ""
        mn = re.search(r"(?:^|\s)(\d+(?:\([a-z]\))?)\s*$", name_note, flags=re.I)
        if mn and (len(name_note) - mn.start()) <= 6:
            note = mn.group(1).strip(); name = name_note[:mn.start()].strip()
        else:
            name = name_note.strip()
        rows.append("|".join([code, name, note, end, begin]))
    return "\n".join(rows)

# ====== Số & Scoring & Auto-route ======
def _is_amount(x: str) -> bool:
    s = (x or "").strip()
    if not s:
        return False
    if re.fullmatch(r"-?\d{1,3}(?:\.\d{3})*", s):
        return True
    return re.fullmatch(r"-?\d{4,}", s) is not None

# ===== Narrator cho bảng var-cols (không ép 5 cột) =====
_NUM_FULL = re.compile(r"-?\d{1,3}(?:\.\d{3})+|-?\d{4,}")
# ===== Sanity-check: var-cols có thực sự là bảng? =====
def _is_var_table(pipe_text: str, min_rows: int = 4, min_num_ratio: float = 0.30) -> bool:
    """
    Trả True nếu pipe N-cột có đủ số dòng & mật độ số.
    - Ít nhất min_rows dòng dữ liệu (không tính header).
    - Tỷ lệ dòng có >=1 số (END/BEGIN hay số bất kỳ) >= min_num_ratio.
    - Loại trừ các trang 'Ý kiến kiểm toán', 'auditor's opinion', ...
    """
    if not pipe_text:
        return False
    lines = [ln for ln in pipe_text.splitlines() if ln.strip()]
    if len(lines) <= 1:
        return False
    body = lines[1:]

    low_all = " ".join(lines).lower()
    if ("ý kiến của kiểm toán" in low_all or "kien toan vien" in low_all
        or "auditor" in low_all and "opinion" in low_all):
        return False

    num_lines = 0
    for ln in body:
        parts = [p.strip() for p in ln.split("|")]
        if any(_NUM_FULL.fullmatch(p or "") for p in parts):
            num_lines += 1
    rows_ok = len(body) >= min_rows
    ratio_ok = (num_lines / max(1, len(body))) >= min_num_ratio
    return rows_ok and ratio_ok

def _parse_varcols_rows(pipe_text: str):
    """Chuyển pipe N-cột thành list hàng {ma, chi, end, start} bằng heuristic:
       - Lấy 2 cột số ngoài cùng bên phải làm END/BEGIN (nếu có).
       - CODE = token đầu nếu trông như '###' hoặc '###.#'.
       - NAME = phần còn lại gộp lại.
    """
    if not pipe_text: return []
    lines = [ln for ln in pipe_text.splitlines() if ln.strip()]
    if len(lines) <= 1: return []
    rows = []
    for ln in lines[1:]:
        parts = [p.strip() for p in ln.split("|")]
        if not any(parts):
            continue
        idx_num = [i for i, p in enumerate(parts) if _NUM_FULL.fullmatch(p or "")]
        end = parts[idx_num[-2]] if len(idx_num) >= 2 else ""
        begin = parts[idx_num[-1]] if len(idx_num) >= 1 else ""
        left = [parts[i] for i in range(len(parts)) if i not in idx_num]
        code = ""
        if left and re.fullmatch(r"\d{3}(?:\.\d+)?", left[0] or ""):
            code = left[0]
            left = left[1:]
        name = " ".join([t for t in left if t]).strip()
        rows.append({"ma": code, "chi": name, "end": end, "start": begin})
    return rows

def narrator_varcols(pipe_text: str) -> str:
    rows = _parse_varcols_rows(pipe_text)
    out = []
    for r in rows:
        # bỏ dòng hoàn toàn không có mã và cũng không có số
        if not (r.get("ma") or r.get("end") or r.get("start")):
            continue
        title = f"[{r['ma']}] {r['chi']}" if r.get("ma") else (r.get("chi") or "")
        if not title.strip():
            continue
        end = (r.get("end") or "").strip()
        start = (r.get("start") or "").strip()
        out.append(f"- {title} — Cuối năm: {end or '∅'}; Đầu năm: {start or '∅'}")
    return "\n".join(out)





def parse_pipe_to_rows(pipe_text: str) -> List[Dict[str,str]]:
    rows = []
    for ln in (pipe_text or "").splitlines():
        parts = [p.strip() for p in ln.split("|")]
        if len(parts) < 5: continue
        code,name,note,end,begin = parts[:5]
        rows.append({"ma":code, "chi":name, "tm":note, "end":end, "start":begin})
    return rows

def score_table_quality(rows: List[Dict[str,str]]) -> Dict[str,float]:
    if not rows:
        return {"row_coverage":0.0,"missing_amount_ratio":1.0,"dup_amount_ratio":0.0,"valid_code_ratio":0.0,"score":0.0}
    covered = sum(1 for r in rows if r.get("ma") or r.get("chi"))
    row_coverage = covered / max(1,len(rows))
    miss = sum(1 for r in rows if not (_is_amount(r.get("end","")) and _is_amount(r.get("start",""))))
    missing_amount_ratio = miss / max(1,len(rows))
    dup = sum(1 for r in rows if r.get("end") and r.get("start") and r["end"] == r["start"])
    dup_amount_ratio = dup / max(1,len(rows))
    valid_code_ratio = sum(1 for r in rows if CODE_3.match((r.get("ma","") or "").strip()))/max(1,len(rows))
    score = row_coverage - 0.6*missing_amount_ratio - 0.3*dup_amount_ratio + 0.2*valid_code_ratio
    return {"row_coverage":row_coverage,"missing_amount_ratio":missing_amount_ratio,"dup_amount_ratio":dup_amount_ratio,"valid_code_ratio":valid_code_ratio,"score":score}

def need_paddle_fallback(metrics: Dict[str,float]) -> bool:
    vio = 0
    if metrics.get("row_coverage",0.0) < 0.72: vio += 1
    if metrics.get("missing_amount_ratio",1.0) > 0.55: vio += 1
    if metrics.get("valid_code_ratio",0.0) < 0.55: vio += 1
    if (metrics.get("missing_amount_ratio",1.0) > 0.60 and 
        metrics.get("valid_code_ratio",0.0) < 0.50):
        vio += 1
    return vio >= 2

def split_glued_amounts(s: str) -> Tuple[str, str]:
    if not s:
        return "", ""
    hits = re.findall(r"\d{1,3}(?:\.\d{3})+|\d{4,}", s)
    if len(hits) >= 2:
        left = normalize_vn_amount(hits[-2])
        right = normalize_vn_amount(hits[-1])
        return left, right
    return "", ""

# ====== Validator gắt & Narrator ======
def yaml_table_clean_rows(rows: List[Dict[str,str]], yaml_rules: Dict[str,Any]) -> List[Dict[str,str]]:
    name_map = (yaml_rules.get("name_alias") or {})
    drop_contains = (yaml_rules.get("drop_if_name_contains") or [])
    out=[]
    for r in rows:
        ma = (r.get("ma","") or "").strip()
        chi= (r.get("chi","") or "").strip()
        tm = (r.get("tm","") or "").strip()
        end= normalize_vn_amount((r.get("end","") or ""))
        start= normalize_vn_amount((r.get("start","") or ""))
        low = _strip_vn_accents(chi)
        if any(tok in low for tok in drop_contains): continue
        for k,v in name_map.items():
            if k in low: chi = v
        out.append({"ma":ma,"chi":chi,"tm":tm,"end":end,"start":start})
    return out

def validate_and_autofix_rows(rows: List[Dict[str,str]]) -> Tuple[List[Dict[str,str]], Dict[str,Any]]:
    issues=[]
    for r in rows:
        if not r["end"] and not r["start"]:
            e2, s2 = split_glued_amounts((r.get("end","") or "") + " " + (r.get("start","") or ""))
            if e2 or s2:
                r["end"], r["start"] = e2, s2
        if r["end"] and not _is_amount(r["end"]): r["end"]=""
        if r["start"] and not _is_amount(r["start"]): r["start"]=""
    metrics = score_table_quality(rows)
    if metrics["missing_amount_ratio"] > 0.60:
        issues.append({"type":"missing_amount_high","ratio":metrics["missing_amount_ratio"]})
    if metrics["dup_amount_ratio"] > 0.15:
        issues.append({"type":"dup_amount_high","ratio":metrics["dup_amount_ratio"]})
    if metrics["valid_code_ratio"] < 0.50:
        issues.append({"type":"invalid_code_ratio","ratio":metrics["valid_code_ratio"]})
    # merge dòng chi tiếp diễn (không mã/không số)
    fused=[]
    for r in rows:
        if (fused and not r["ma"] and not r["tm"] and not r["end"] and not r["start"]
            and r["chi"] and len(r["chi"]) <= 48):
            fused[-1]["chi"] = (fused[-1]["chi"] + " " + r["chi"]).strip()
        else:
            fused.append(r)
    return fused, {"metrics":metrics, "issues":issues}

# === CROSS-CHECK HELPERS ===
def _rows_to_amount_map(rows):
    import re
    def to_int(s):
        if not s: return None
        digits = re.sub(r"[^\d-]", "", s.replace(".", ""))
        return int(digits) if digits not in ("", "-") else None
    m_end, m_start = {}, {}
    for r in rows:
        code = (r.get("ma") or "").strip()
        if not code:
            continue
        key = code.split(".")[0]
        e = to_int(r.get("end"))
        s = to_int(r.get("start"))
        if e is not None:   m_end[key]   = e
        if s is not None:   m_start[key] = s
    return m_end, m_start

def _eval_eq(expr, amap):
    """
    Đánh giá biểu thức kiểu '270 = 100 + 200'.
    - Ưu tiên coi token là MÃ (tra theo amap); nếu không có, mặc định 0.
    - Chỉ coi là hằng số khi là số 1–2 chữ số (ví dụ 1, 2, 10).
    """
    import re
    expr = (expr or "").strip()
    if "=" not in expr:
        return True, None, None

    left, right = [x.strip() for x in expr.split("=", 1)]

    def as_value(tok: str) -> int:
        tok = tok.strip()
        key = tok.split(".")[0]  # gom 210.1 -> 210
        # 1) nếu có trong map, trả về số tiền
        if key in amap:
            return amap[key]
        # 2) nếu thật sự là hằng số nhỏ (1–2 chữ số) thì cho là literal
        if re.fullmatch(r"\d{1,2}", tok):
            return int(tok)
        # 3) còn lại coi là mã nhưng thiếu dữ liệu → 0
        return 0

    def eval_side(side: str) -> int:
        total, sign = 0, +1
        for tok in re.findall(r"[+-]|\d+(?:\.\d+)?", side):
            if tok in ["+", "-"]:
                sign = +1 if tok == "+" else -1
            else:
                total += sign * as_value(tok)
        return total

    lhs = as_value(left)
    rhs = eval_side(right)
    return (lhs == rhs), lhs, rhs


def run_crosschecks(rows, yaml_rules):
    formulas = ((yaml_rules.get("globals") or {}).get("cross_formulas") or [])
    m_end, m_start = _rows_to_amount_map(rows)
    results = []
    for f in formulas:
        ok_end,  lhs_end,  rhs_end   = _eval_eq(f.get("end",""),   m_end)   if f.get("end")   else (True, None, None)
        ok_start,lhs_start,rhs_start = _eval_eq(f.get("start",""), m_start) if f.get("start") else (True, None, None)
        results.append({
            "name": f.get("name"),
            "end":   {"ok": ok_end,   "lhs": lhs_end,   "rhs": rhs_end},
            "start": {"ok": ok_start, "lhs": lhs_start, "rhs": rhs_start},
        })
    summary_ok = all((r["end"]["ok"] and r["start"]["ok"]) for r in results) if results else True
    return summary_ok, results

def narrator_rows(rows):
    def _numstr(x):
        s = str(x or "").strip().lower()
        return "" if s in ("nan","none") else str(x or "")
    out=[]
    for r in rows:
        chi = (r.get("chi","") or "").strip()
        if not chi or len(chi) < 2:
            continue
        ma = (r.get("ma","") or "").strip()
        tm = (r.get("tm","") or "").strip()
        end, start = _numstr(r.get("end","")), _numstr(r.get("start",""))
        parts = []
        title = f"[{ma}] {chi}" if ma else chi
        parts.append(title)
        if end or start:
            parts.append(f"Cuối năm: {end or '∅'}; Đầu năm: {start or '∅'}")
        if tm:
            parts.append(f"TM: {tm}")
        out.append("- " + " — ".join(parts))
    return "\n".join(out)

# ====== TABLE build pipelines ======
def build_table_tsv(pil: Image.Image, y_tol: int, ocr_lang: str) -> Tuple[str, Dict[str,float]]:
    from sklearn.metrics import silhouette_score

    bgr = cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    tsv = pytesseract.image_to_data(bgr, lang=ocr_lang, config=OCR_CFG_TSV, output_type=TessOutput.DICT)
    df = _tsv_dict_to_df(tsv)
    if df.empty:
        return "", {"score": 0.0}

    rows = _cluster_rows(df, y_tol=y_tol)
    var_rows: List[List[str]] = []
    max_cols_seen = 0

    for row_df in rows:
        row_df = row_df[row_df["text"].astype(str).str.strip().astype(bool)]
        if row_df.empty:
            continue
        X = row_df[["cx"]].to_numpy(dtype=float)
        if len(X) <= 2:
            toks = [str(t).strip() for t in row_df.sort_values("cx")["text"].tolist()]
            var_rows.append([" ".join(toks)])
            max_cols_seen = max(max_cols_seen, 1)
            continue

        best_lab, best_score = None, -1.0
        for k in range(2, min(10, len(X)) + 1):
            try:
                km = KMeans(n_clusters=k, n_init="auto", random_state=0).fit(X)
                lab = km.labels_
                if len(set(lab)) < 2:
                    continue
                s = silhouette_score(X, lab)
                if s > best_score:
                    best_score, best_lab = s, lab
            except Exception:
                continue

        if best_lab is None:
            toks = [str(t).strip() for t in row_df.sort_values("cx")["text"].tolist()]
            var_rows.append([" ".join(toks)])
            max_cols_seen = max(max_cols_seen, 1)
            continue

        row_df2 = row_df.copy(); row_df2["col_id"] = best_lab
        centers = row_df2.groupby("col_id")["cx"].mean().sort_values()
        order = {cid:i for i,cid in enumerate(centers.index.tolist())}
        row_df2["col_id"] = row_df2["col_id"].map(order)

        buckets = {cid: [] for cid in sorted(row_df2["col_id"].unique())}
        for _, t in row_df2.sort_values(["col_id","cx"]).iterrows():
            s = _norm_word(str(t["text"]))
            if s:
                buckets[int(t["col_id"])].append(s)

        cells = [" ".join(buckets[cid]).strip() for cid in sorted(buckets.keys())]
        var_rows.append(cells)
        max_cols_seen = max(max_cols_seen, len(cells))

    header = [f"C{i+1}" for i in range(max_cols_seen)]
    lines = [" | ".join(header)]
    for cells in var_rows:
        if len(cells) < max_cols_seen:
            cells = cells + [""]*(max_cols_seen - len(cells))
        lines.append(" | ".join(cells))

    pipe = "\n".join(lines)
    return pipe, {"score": 1.0}


def build_table_paddle(pil: Image.Image, paddle_lang: str, paddle_gpu: bool) -> Tuple[str, Dict[str,float]]:
    pipe = paddle_table_to_pipe(pil, lang=paddle_lang, use_gpu=paddle_gpu) or ""
    rows_struct = parse_pipe_to_rows(pipe)
    metrics = score_table_quality(rows_struct)
    return pipe, metrics

# ====== QUY TRÌNH 1 TRANG (VAR-COLS, KHÔNG ÉP 5 CỘT) ======
def process_page(
    pil: Image.Image, yaml_table: dict, yaml_text: dict, ocr_lang: str,
    use_gpt: bool, gpt_table_mode: str, gpt_model: str, gpt_temp: float,
    log_gpt: bool, gpt_scope: str = "table_only",
    do_autofix: bool = True, force_table: bool = False,
    rebuild_table: str = "auto", y_tol: int = 8,
    ocr_engine: str = "auto", table_engine: str = "auto",
    paddle_lang: str = "vi", paddle_gpu: bool = False,
    narrator_on: bool = True, table_format: str = "pipe"
) -> Tuple[str, str, dict, List[Tuple[str,str]]]:
    """
    Trả về: (block_type, block_text, block_meta, extra_blocks)
    - block_text: TEXT hoặc TABLE (bảng N cột thực tế, header: COL1|COL2|...|COLN)
    - extra_blocks: có thể thêm block khác (ví dụ narr/cross nếu sau này cần)
    """
    extra_blocks: List[Tuple[str, str]] = []
    gpt_used = False

    # OCR chọn engine (giữ nguyên logic cũ; text detection/meta vẫn cần)
    try:
        if ocr_engine == "paddle" or (ocr_engine == "auto" and _HAS_PADDLE and False):
            ocr_txt, meta_json = ocr_image_text_paddle(pil, lang=paddle_lang, use_gpu=paddle_gpu)
        else:
            ocr_txt, meta_json = ocr_image_text(pil, lang=ocr_lang)
    except Exception as e:
        print(f"[WARN] OCR engine failed ({ocr_engine}): {e} → fallback Tesseract")
        ocr_txt, meta_json = ocr_image_text(pil, lang=ocr_lang)

    meta = json.loads(meta_json) if meta_json else {}
    meta["_table_format"] = table_format  # vẫn giữ trường này
    meta["table_schema"] = "varcols"      # đánh dấu hiện đang dùng N cột linh hoạt

    # Phát hiện ROI bảng để tách TEXT/TABLE (giữ nguyên)
    extra_blocks_mixed: List[Tuple[str, str]] = []
    try:
        _bgr0 = cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)
        rois = _find_table_rois(_bgr0)
    except Exception:
        rois = []

    # log diện tích bảng (giữ nguyên)
    try:
        H, W = _bgr0.shape[:2]
        page_area = float(H * W)
        roi_area = sum((x2 - x1) * (y2 - y1) for (x1, y1, x2, y2) in rois) if rois else 0.0
        meta["roi_area_ratio"] = round(roi_area / max(1.0, page_area), 4)
    except Exception:
        meta["roi_area_ratio"] = None

    if rois and not force_table:
        # TEXT không bảng (giữ nguyên)
        try:
            pil_text_only = _mask_out_rois(pil, rois)
            txt_only, _meta_text = ocr_image_text(pil_text_only, lang=ocr_lang)
            block_text_text = apply_yaml_clean(txt_only, yaml_text, mode="text").strip()
        except Exception:
            block_text_text = ""

        meta.setdefault("roi_padded", [])
        meta.setdefault("gpt_used_roi", [])
        meta.setdefault("roi_table_metrics", [])

        for (x1, y1, x2, y2) in rois:
            gl = (yaml_table.get("globals") or {})
            pad_top = int(gl.get("roi_pad_top", 120))
            pad_lr  = int(gl.get("roi_pad_lr", 8))

            y1_p = max(0, int(y1) - pad_top)
            x1_p = max(0, int(x1) - pad_lr)
            x2_p = min(pil.width,  int(x2) + pad_lr)
            y2_p = min(pil.height, int(y2))

            pil_roi = pil.crop((x1_p, y1_p, x2_p, y2_p))
            meta["roi_padded"].append([x1_p, y1_p, x2_p, y2_p])

            # === BẢNG VAR-COLS (không ép 5 cột) — dùng làm block cho ROI ===
            var_pipe = build_generic_table_from_tsv(
                pil_roi, y_tol=y_tol, ocr_lang=ocr_lang, max_cols=12
            ) or ""

            # Bỏ ROI nếu không phải bảng "thật"
            if not _is_var_table(var_pipe):
                continue

            # GPT (tuỳ chọn) — 'generic' để không ép schema tài chính
            gpt_used_roi = False
            if (gpt_scope in ("table_only", "all")) and use_gpt and _HAS_GPT_ENHANCER and var_pipe.strip():
                try:
                    block2 = enhance_table_with_gpt(
                        table_text_cleaned=var_pipe,
                        image_pil=pil_roi,
                        meta={"company_hint": meta.get("company"), "period_hint": meta.get("period")},
                        mode="generic",
                        model=gpt_model, temperature=gpt_temp, log_diag=log_gpt,
                    )
                    if isinstance(block2, str) and block2.strip():
                        var_pipe = block2.strip()
                        gpt_used_roi = True
                except Exception as e:
                    meta.setdefault("gpt_errors_roi", []).append(str(e))
            meta["gpt_used_roi"].append(bool(gpt_used_roi))

            # Metrics đơn giản: rows/cols (không dựa 5 cột)
            header = (var_pipe.splitlines() or [""])[0] if var_pipe.strip() else ""
            ncols = len([c for c in header.split("|") if c.strip()]) if header else 0
            nrows = max(0, (len(var_pipe.splitlines()) - 1)) if var_pipe.strip() else 0

            meta["roi_table_metrics"].append({
                "roi": [x1_p, y1_p, x2_p, y2_p],  # dùng y2_p cho nhất quán
                "route": "tsv-varcols",
                "metrics": {"rows": nrows, "cols": ncols}
            })

            # Gắn TABLE + narrator cho ROI
            extra_blocks_mixed.append(("TABLE", var_pipe))
            narr = narrator_varcols(var_pipe)
            if narr.strip():
                extra_blocks_mixed.append(("TABLE→ROW-NARR", narr))


        if extra_blocks_mixed:
            meta.update({
                "mixed_page": True,
                "roi_count": len(rois),
                "engine": meta.get("engine"),
            })
            # trả TEXT làm block chính, bảng ROI là extra blocks
            return "TEXT", block_text_text, meta, extra_blocks_mixed

    # --- Không tách ROI: quyết định TABLE cho cả trang? (giữ nguyên phát hiện, đổi dựng) ---
    try:
        is_tbl = is_table_page(ocr_txt, yaml_table) or force_table
    except TypeError:
        is_tbl = is_table_page(ocr_txt, {}) or force_table

    # Fallback thử dựng bảng dù detector nói không phải — SIẾT CHẶT, CHỈ CHO BẢNG "THẬT"
    if not is_tbl:
        var_probe = build_generic_table_from_tsv(
            pil, y_tol=y_tol, ocr_lang=ocr_lang, max_cols=12
        ) or ""

        # Chỉ nhận là bảng nếu đủ chuẩn (rows>=6 & mật độ dòng có số >= 0.50)
        if _is_var_table(var_probe, min_rows=6, min_num_ratio=0.50):
            # (tùy chọn) GPT generic
            gpt_used = False
            if (gpt_scope in ("table_only","all")) and use_gpt and _HAS_GPT_ENHANCER:
                try:
                    block2 = enhance_table_with_gpt(
                        table_text_cleaned=var_probe,
                        image_pil=pil,
                        meta={"company_hint": meta.get("company"), "period_hint": meta.get("period")},
                        mode="generic", model=gpt_model, temperature=gpt_temp, log_diag=log_gpt,
                    )
                    if isinstance(block2, str) and block2.strip():
                        var_probe = block2.strip()
                        gpt_used = True
                except Exception as e:
                    meta["gpt_error"] = str(e)

            # luôn bổ sung TEXT mô tả đi kèm
            try:
                block_text_text = apply_yaml_clean(ocr_txt, yaml_text, mode="text").strip()
            except Exception:
                block_text_text = (ocr_txt or "").strip()
            if block_text_text:
                extra_blocks.append(("TEXT", block_text_text))

            narr_full = narrator_varcols(var_probe)
            if narr_full.strip():
                extra_blocks.append(("TABLE→ROW-NARR", narr_full))

            # metrics
            header = (var_probe.splitlines() or [""])[0]
            ncols = len([c for c in header.split("|") if c.strip()]) if header else 0
            nrows = max(0, len(var_probe.splitlines()) - 1)

            block_text = var_probe
            meta.update({
                "gpt_used": gpt_used,
                "table_route": "tsv-varcols(fallback)",
                "table_metrics": {"rows": nrows, "cols": ncols},
                "text_sha1": _sha1(block_text),
                "engine": meta.get("engine"),
                "table_format": "pipe-varcols",
            })
            return "TABLE", block_text, meta, extra_blocks

        # KHÔNG đạt chuẩn bảng → coi là TEXT thuần
        try:
            block = apply_yaml_clean(ocr_txt, yaml_text, mode="text").strip()
        except Exception:
            block = (ocr_txt or "").strip()
        meta["gpt_used"] = False
        meta["text_sha1"] = _sha1(block)
        return "TEXT", block, meta, []

    # === is_tbl == True: Dựng bằng best-of-two (TSV var-cols vs Paddle) ===
    if is_tbl:
        # 1) TSV var-cols
        var_pipe_full = build_generic_table_from_tsv(
            pil, y_tol=y_tol, ocr_lang=ocr_lang, max_cols=12
        ) or ""

        # 2) Paddle (nếu có) để so sánh
        paddle_pipe = ""
        if _HAS_PADDLE:
            try:
                paddle_pipe = paddle_table_to_pipe(pil, lang=paddle_lang, use_gpu=paddle_gpu) or ""
            except Exception as e:
                meta["paddle_table_error"] = str(e)

        # 3) Chọn “bảng tốt hơn” dựa trên _is_var_table (mật độ số) và số cột/hàng
        def _tbl_score(p):
            if not _is_var_table(p, min_rows=6, min_num_ratio=0.45):
                return (0, 0)
            header = (p.splitlines() or [""])[0]
            ncols = len([c for c in header.split("|") if c.strip()]) if header else 0
            nrows = max(0, len(p.splitlines()) - 1)
            return (ncols, nrows)

        cand = [("tsv", var_pipe_full), ("paddle", paddle_pipe)]
        cand = [(tag, p, _tbl_score(p)) for tag,p in cand if (p or "").strip()]
        cand.sort(key=lambda t: t[2], reverse=True)  # ưu tiên nhiều cột/hàng hơn
        if cand:
            route, var_pipe_full, (ncols, nrows) = cand[0]
        else:
            # không ứng viên đủ chuẩn → rơi về TEXT thay vì nhồi TABLE rác
            try:
                block = apply_yaml_clean(ocr_txt, yaml_text, mode="text").strip()
            except Exception:
                block = (ocr_txt or "").strip()
            meta["gpt_used"] = False
            meta["text_sha1"] = _sha1(block)
            return "TEXT", block, meta, []

        # 4) (tùy chọn) GPT generic
        gpt_used = False
        if (gpt_scope in ("table_only","all")) and use_gpt and _HAS_GPT_ENHANCER and var_pipe_full.strip():
            try:
                block2 = enhance_table_with_gpt(
                    table_text_cleaned=var_pipe_full,
                    image_pil=pil,
                    meta={"company_hint": meta.get("company"), "period_hint": meta.get("period")},
                    mode="generic", model=gpt_model, temperature=gpt_temp, log_diag=log_gpt,
                )
                if isinstance(block2, str) and block2.strip():
                    var_pipe_full = block2.strip()
                    gpt_used = True
            except Exception as e:
                meta["gpt_error"] = str(e)

        # 5) Luôn thêm TEXT mô tả đi kèm bảng
        try:
            block_text_text = apply_yaml_clean(ocr_txt, yaml_text, mode="text").strip()
        except Exception:
            block_text_text = (ocr_txt or "").strip()
        if block_text_text:
            extra_blocks.append(("TEXT", block_text_text))

        narr_full = narrator_varcols(var_pipe_full)
        if narr_full.strip():
            extra_blocks.append(("TABLE→ROW-NARR", narr_full))

        block_text = var_pipe_full
        meta.update({
            "gpt_used": gpt_used,
            "table_route": f"{route}-varcols",
            "table_metrics": {"rows": nrows, "cols": ncols},
            "text_sha1": _sha1(block_text),
            "engine": meta.get("engine"),
            "table_format": "pipe-varcols",
        })
        return "TABLE", block_text, meta, extra_blocks

    # --- TEXT nhánh (giữ nguyên) ---
    try:
        block = apply_yaml_clean(ocr_txt, yaml_text, mode="text").strip()
    except Exception:
        block = (ocr_txt or "").strip()
    meta["gpt_used"] = False
    meta["text_sha1"] = _sha1(block)
    return "TEXT", block, meta, []


# ====== MIRROR OUTPUT PATH ======
def make_output_paths(input_root: str, output_root: str, file_path: str) -> Tuple[str,str]:
    rel = os.path.relpath(file_path, start=input_root)
    rel_dir = os.path.dirname(rel)
    stem = os.path.splitext(os.path.basename(file_path))[0]
    out_dir = os.path.join(output_root, rel_dir); ensure_dir(out_dir)
    return (os.path.join(out_dir, f"{stem}_text.txt"),
            os.path.join(out_dir, f"{stem}_meta.json"))

# ====== MAIN PIPELINE FOR ONE FILE ======
def process_one_file(file_path: str, input_root: str, output_root: str,
                     yaml_table: dict, yaml_text: dict,
                     ocr_lang: str, split_debug: bool, start_page: Optional[int], end_page: Optional[int],
                     use_gpt: bool, gpt_table_mode: str, gpt_model: str, gpt_temp: float, log_gpt: bool,
                     gpt_scope: str = "table_only",
                     no_autofix: bool = False,
                     force_table: bool = False,
                     rebuild_table: str = "auto", y_tol: int = 8,
                     ocr_engine: str = "auto", table_engine: str = "auto",
                     paddle_lang: str = "vi", paddle_gpu: bool = False,
                     narrator_on: bool = True, dpi: int = 360,
                     table_format: str = "pipe"
                     ) -> None:

    print(f"📄 Input: {file_path}")
    out_txt, out_meta = make_output_paths(input_root, output_root, file_path)

    blocks: List[str] = []
    blocks_text_only: List[str] = []
    blocks_table_only: List[str] = []
    page_metas: List[dict] = []

    total_pages = 0
    for page_idx, pil in iter_pages(file_path, dpi=dpi):
        if start_page is not None and page_idx + 1 < start_page:
            continue
        if end_page is not None and page_idx + 1 > end_page:
            continue
        total_pages += 1

        btype, block, meta, extra_blocks = process_page(
            pil, yaml_table, yaml_text, ocr_lang,
            use_gpt=use_gpt and _HAS_GPT_ENHANCER,
            gpt_table_mode=gpt_table_mode, gpt_model=gpt_model, gpt_temp=gpt_temp, log_gpt=log_gpt,
            gpt_scope=gpt_scope,
            do_autofix=(not no_autofix), force_table=force_table,
            rebuild_table=rebuild_table, y_tol=y_tol,
            ocr_engine=ocr_engine, table_engine=table_engine,
            paddle_lang=paddle_lang, paddle_gpu=paddle_gpu,
            narrator_on=narrator_on,
            table_format=table_format
        )

        header = f"### [PAGE {page_idx+1:02d}] [{btype}]"
        blocks.append(f"{header}\n{block}\n")
        if btype == "TEXT":
            blocks_text_only.append(f"{header}\n{block}\n")
        else:
            blocks_table_only.append(f"{header}\n{block}\n")

        for (bt2, bl2) in extra_blocks:
            header2 = f"### [PAGE {page_idx+1:02d}] [{bt2}]"
            blocks.append(f"{header2}\n{bl2}\n")
            blocks_table_only.append(f"{header2}\n{bl2}\n")

        meta["page"] = page_idx + 1
        meta["block_type"] = btype
        page_metas.append(meta)

    final_text = "\n".join(blocks).strip()
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(final_text)
    print(f"📝 Wrote TXT: {out_txt}")

    try:
        import re
        preview = re.sub(r"\s+", " ", final_text).strip()
        print("   [OUT PREVIEW]", (preview[:100] + ("…" if len(preview) > 100 else "")))
    except Exception as e:
        print(f"   [OUT PREVIEW error: {e}]")

    meta_obj = {
        "file": os.path.basename(file_path),
        "source_path": os.path.abspath(file_path),
        "total_pages_processed": total_pages,
        "language_hint": page_metas[0]["language"] if page_metas else None,
        "period": page_metas[0]["period"] if page_metas else None,
        "company": page_metas[0]["company"] if page_metas else None,
        "blocks": page_metas,
        "ocr_lang": ocr_lang,
    }
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta_obj, f, ensure_ascii=False, indent=2)
    print(f"🧾 Wrote META: {out_meta}")

    if split_debug:
        stem = os.path.splitext(out_txt)[0]
        if blocks_text_only:
            with open(stem + "_TEXT.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(blocks_text_only).strip())
        if blocks_table_only:
            with open(stem + "_TABLE.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(blocks_table_only).strip())
        print("🔎 Split-debug files written.")

# ====== CLI ======
def build_argparser():
    p = argparse.ArgumentParser("P1A (GPT) — Scan gốc → TXT (TEXT/TABLE) + META (per-file) [Hybrid Auto-Route]")
    p.add_argument("--input-root", type=str, default=INPUT_ROOT_DEFAULT)
    p.add_argument("--out-root",   type=str, default=OUTPUT_ROOT_DEFAULT)
    p.add_argument("--yaml-table", type=str, default=YAML_TABLE_DEFAULT)
    p.add_argument("--yaml-text",  type=str, default=YAML_TEXT_DEFAULT)
    p.add_argument("--ocr-lang",   type=str, default=OCR_LANG_DEFAULT)
    p.add_argument("--split-debug", action="store_true")
    p.add_argument("--start", type=int, default=None)
    p.add_argument("--end",   type=int, default=None)
    # GPT toggles
    p.add_argument("--no-gpt", action="store_true")
    p.add_argument("--gpt-table-mode", choices=["financial","generic","auto"], default="financial")
    p.add_argument("--gpt-model", type=str, default="gpt-4o-mini")
    p.add_argument("--gpt-temp", type=float, default=0.0)
    p.add_argument("--log-gpt", action="store_true")
    # Validator toggle
    p.add_argument("--no-autofix", action="store_true")
    p.add_argument("--force-table", action="store_true")
    # Rebuild flags
    p.add_argument("--rebuild-table", choices=["auto","none","force"], default="auto")
    p.add_argument("--y-tol", type=int, default=18)
    # OCR/Table engine defaults
    p.add_argument("--ocr-engine", choices=["auto","tesseract","paddle"], default="tesseract")
    p.add_argument("--table-engine", choices=["auto","tsv","paddle"], default="tsv")
    p.add_argument("--paddle-lang", type=str, default="vi")
    p.add_argument("--paddle-gpu", action="store_true")
    # Narrator & DPI
    p.add_argument("--narrator", choices=["y","n"], default="y")
    p.add_argument("--dpi", type=int, default=360, help="DPI render PDF/DOCX (khuyên 360–420)")
    # Định dạng xuất bảng
    p.add_argument("--table-format", choices=["ascii","pipe","json"], default="pipe",
                   help="Định dạng xuất bảng: ascii (khung), pipe (CODE|NAME|NOTE|END|BEGIN), json (list rows)")
    # Phạm vi GPT
    p.add_argument("--gpt-scope", choices=["table_only","all","none"], default="table_only",
                   help="Phạm vi dùng GPT: table_only (chỉ bảng), all (cả text & bảng), none (tắt GPT)")
    return p

def main():
    args = build_argparser().parse_args()

    # --- Summary cấu hình & RAW/YAML toggle ---
    print("========== P1A RUN CONFIG ==========")
    print(f"[INPUT ] input-root : {os.path.abspath(args.input_root)}")
    print(f"[OUTPUT] out-root   : {os.path.abspath(args.out_root)}")
    print(f"[ENGINE] OCR={args.ocr_engine} | TABLE={args.table_engine} | DPI={args.dpi}")
    print(f"[FORMAT] table-format={args.table_format} | narrator={'ON' if args.narrator=='y' else 'OFF'}")
    print(f"[GPT   ] enabled={('NO' if args.no_gpt else 'YES')} | mode={args.gpt_table_mode} | model={args.gpt_model}")
    print(f"[YAML ] Trạng thái: {'ĐANG TẮT (RAW MODE)' if P1A_RAW_MODE else 'ĐANG BẬT (dùng YAML)'}")
    if not P1A_RAW_MODE:
        print(f"[YAML ] yaml-table : {args.yaml_table}")
        print(f"[YAML ] yaml-text  : {args.yaml_text}")
    print("====================================")

    # --- Nạp YAML theo trạng thái RAW ---
    if P1A_RAW_MODE:
        print("🔧 RAW_MODE=ON → Tắt YAML table/text & prefilter/mapping/validator/cross-check")
        yaml_table, yaml_text = {}, {}
    else:
        try:
            with open(args.yaml_table, "r", encoding="utf-8") as f:
                yaml_table = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"⚠️ Không đọc được YAML bảng: {e}")
            yaml_table = {}
        try:
            with open(args.yaml_text, "r", encoding="utf-8") as f:
                yaml_text = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"⚠️ Không đọc được YAML text: {e}")
            yaml_text = {}

    # --- Lọc file input ---
    exts = (".pdf", ".docx", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
    files = [
        p for p in glob.glob(os.path.join(args.input_root, "**", "*"), recursive=True)
        if os.path.isfile(p) and os.path.splitext(p)[1].lower() in exts
    ]
    print(f"[DEBUG] input-root = {args.input_root}")
    print(f"[DEBUG] out-root   = {args.out_root}  (📁 tất cả file xuất sẽ nằm dưới thư mục này)")
    print(f"[DEBUG] found {len(files)} file(s)")
    if not files:
        print(f"⚠️ Không tìm thấy file hợp lệ dưới: {args.input_root}")
        return

    # --- Xử lý từng file ---
    for fp in tqdm(files, desc="Processing files"):
        try:
            process_one_file(
                file_path=fp,
                input_root=args.input_root,
                output_root=args.out_root,
                yaml_table=yaml_table,
                yaml_text=yaml_text,
                ocr_lang=args.ocr_lang,
                split_debug=args.split_debug,
                start_page=args.start,
                end_page=args.end,
                use_gpt=(not args.no_gpt),
                gpt_table_mode=args.gpt_table_mode,
                gpt_model=args.gpt_model,
                gpt_temp=args.gpt_temp,
                log_gpt=args.log_gpt,
                gpt_scope=args.gpt_scope,
                no_autofix=args.no_autofix,
                force_table=getattr(args, "force_table", False),
                rebuild_table=args.rebuild_table, y_tol=args.y_tol,
                ocr_engine=args.ocr_engine, table_engine=args.table_engine,
                paddle_lang=args.paddle_lang, paddle_gpu=args.paddle_gpu,
                narrator_on=(args.narrator=="y"),
                dpi=args.dpi,
                table_format=args.table_format,
            )
            # Lưu ý: process_one_file đã in ra đường dẫn TXT và META:
            #  📝 Wrote TXT: <..._text.txt>
            #  🧾 Wrote META: <..._meta.json>
        except Exception as e:
            import traceback
            print(f"❌ Lỗi file: {fp} → {e}")
            traceback.print_exc()

    print("\n✅ Hoàn tất. Mỗi file input sinh ra 1 TXT (TEXT/TABLE + diễn giải dòng) + 1 meta.json.")
    print("   Bật --split-debug để có thêm file _TEXT/_TABLE; dùng --narrator n để tắt diễn giải khi QA.")
    print(f"📌 Toàn bộ output đang nằm dưới thư mục: {os.path.abspath(args.out_root)}")


if __name__ == "__main__":
    main()
