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
OUTPUT_ROOT_DEFAULT = r"D:\1.TLAT\3. ChatBot_project\1_Insurance_Strategy\outputs\p1a_clean10_ocr_bctc_GPT_test"
YAML_TABLE_DEFAULT  = r"D:\1.TLAT\3. ChatBot_project\1_Insurance_Strategy\configs\p1a_clean10_ocr_bctc_table.yaml"
YAML_TEXT_DEFAULT   = r"D:\1.TLAT\3. ChatBot_project\1_Insurance_Strategy\configs\p1a_clean10_ocr_bctc_text.yaml"

# ====== OCR CONFIG ======
OCR_LANG_DEFAULT = "vie+eng"
OCR_CFG_SENTENCE = "--psm 4 -c preserve_interword_spaces=1"
OCR_CFG_TSV      = "--psm 4 -c preserve_interword_spaces=1"

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
    if not text: return False
    if re.search(r"Ý kiến của Ki(ê|e)m toán|Ki(ê|e)m toán viên|Auditor'?s opinion", text, re.I):
        return False
    det = (yaml_table.get("globals") or {}).get("detection_tokens") or {}
    must_any = det.get("must_have_any") or []
    if any(re.search(re.escape(tok), text, re.I) for tok in must_any):
        return True
    code_hits = len(CODE_LINE.findall(text))
    money_lines = sum(1 for ln in text.splitlines() if len(MONEY.findall(ln)) >= 2)
    if code_hits >= 3 and money_lines >= 3:
        return True
    kw = r"(mã\s*số|ma\s*so|chỉ\s*tiêu|chi\s*tieu|thuyết\s*minh|thuyet\s*minh|as\s+at|vnd|B0?1\s*-\s*DNPNT)"
    return re.search(kw, text, re.I) is not None

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

def _anchor_numeric_splits(tsv_df: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
    if tsv_df.empty: return None, None
    H = tsv_df["top"].max() + tsv_df["height"].max()
    cut = 0.35 * H
    head = tsv_df[tsv_df["top"] < cut].copy()
    if head.empty: return None, None

    end_x = []; begin_x = []
    for _, r in head.iterrows():
        txt = _norm_word(str(r["text"]))
        if _HEADER_HINTS_END.search(txt):   end_x.append(r["cx"])
        if _HEADER_HINTS_BEGIN.search(txt): begin_x.append(r["cx"])
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

def _row_is_valid(code, name, end, begin):
    if CODE_3.match((code or "").strip()): return True
    if (end and begin): return True
    return len((name or "").strip()) >= 7

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
            left_tokens, mid_tokens, end_tokens, begin_tokens = [], [], [], []
            for _, t in row_df.sort_values("cx").iterrows():
                cx = float(t["cx"]); s = _norm_word(str(t["text"]))
                s = _clean_name_token(s)
                if not s: continue
                if cx < (split1 - 16):
                    left_tokens.append(s)
                elif cx < (split2 - 16):
                    if _is_numberish(s): end_tokens.append(s)
                    else: mid_tokens.append(s)
                elif cx > (split2 + 16):
                    if _is_numberish(s): begin_tokens.append(s)
                    else: mid_tokens.append(s)
                else:
                    if abs(cx - split1) < abs(cx - split2): end_tokens.append(s)
                    else: begin_tokens.append(s)

            left_txt = " ".join([t for t in left_tokens + mid_tokens if t]).strip()
            parts = [p.strip() for p in re.split(r"(?<!^)\s(?=\d{3}(?:\.\d+)?\b)", left_txt)]
            code = ""; name = left_txt; note = ""
            if parts:
                if re.match(r"^\d{3}(?:\.\d+)?$", parts[0]):
                    code = parts[0]
                    name = " ".join(parts[1:]).strip() if len(parts) > 1 else ""
            m = re.search(r"(?:^|\s)(\d+(?:\([a-z]\))?)\s*$", name, flags=re.I)
            if m and (len(name) - m.start()) <= 6:
                note = m.group(1).strip()
                name = name[:m.start()].strip()

            end_val   = _fix_vn_number(" ".join(end_tokens))
            begin_val = _fix_vn_number(" ".join(begin_tokens))

            if not end_val and not begin_val:
                e2, b2 = _fallback_two_num_by_hist(row_df, page_w=W)
                end_val = _fix_vn_number(e2); begin_val = _fix_vn_number(b2)

            if not _row_is_valid(code, name, end_val, begin_val):
                continue

            cols[0], cols[1], cols[2], cols[3], cols[4] = code, name, note, end_val, begin_val
            table.append(cols)
            continue

        # --- fallback KMeans ---
        row_df2 = _infer_columns(row_df, max_cols=5)
        if row_df2 is None:
            code, name, note = "", " ".join(row_df.sort_values("cx")["text"].astype(str)), ""
            e2, b2 = _fallback_two_num_by_hist(row_df, page_w=W)
            end, begin = _fix_vn_number(e2), _fix_vn_number(b2)
            if not _row_is_valid(code, name, end, begin): continue
            table.append([code, name, note, end, begin])
            continue

        buckets = {cid: [] for cid in sorted(row_df2["col_id"].unique())}
        for _, t in row_df2.iterrows():
            txt = _clean_name_token(_norm_word(str(t["text"])))
            if not txt: continue
            buckets[int(t["col_id"])].append(txt)

        cols_temp = [""] * len(buckets)
        for cid in sorted(buckets.keys()):
            cell = " ".join(buckets[cid]).strip()
            if cid >= (len(buckets)-2) and not _is_numberish(cell):
                if len(cell) <= 8:   cols_temp[min(2, len(cols_temp)-1)] += (" " + cell).strip()
                else:                cols_temp[min(1, len(cols_temp)-1)] += (" " + cell).strip()
                cell = ""
            cols_temp[cid] = cell

        code, name, note, end, begin = "", "", "", "", ""
        if len(cols_temp) >= 5:
            code, name, note = cols_temp[0], cols_temp[1], cols_temp[2]
            end, begin = _fix_vn_number(cols_temp[-2]), _fix_vn_number(cols_temp[-1])
        else:
            seq = cols_temp + [""]*(5-len(cols_temp))
            code, name, note = seq[0], seq[1], seq[2]
            end, begin = _fix_vn_number(seq[3]), _fix_vn_number(seq[4])

        if not _row_is_valid(code, name, end, begin): continue
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

    # 1) Preprocess cho detector
    bgr_raw = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    bgr = _preprocess_for_paddle(bgr_raw)

    layout_lang = lang if lang in ("en","ch") else "en"
    table_engine = get_ppstructure(lang=layout_lang, use_gpu=use_gpu)
    result = table_engine(bgr)

    # gom các bảng
    tables = []
    table_items = []
    for item in result:
        if item.get("type") == "table":
            table_items.append(item)
            res = item.get("res", {})
            html = res.get("html")
            if not html:
                continue
            try:
                dfs = pd.read_html(html)
                for df in dfs:
                    tables.append(df)
            except Exception:
                continue

    if not tables and not table_items:
        return None

    # 2) Ghép df (html) — text base
    df = None
    for t in tables:
        if df is None:
            df = t
        else:
            try:
                df = pd.concat([df, t], axis=0, ignore_index=True)
            except Exception:
                pass
    if df is None:
        df = pd.DataFrame()

    # 3) Map 5 cột
    def _is_numish_series(series, ncheck=50):
        vals = series.head(ncheck).astype(str).tolist()
        return any(re.search(r"[\d()\-\.,]", v or "") for v in vals)

    if not df.empty:
        cols = [str(c).strip() for c in df.columns]
        df2 = df.copy()
        df2.columns = cols
        if df2.shape[1] >= 5:
            right_num = [c for c in df2.columns if _is_numish_series(df2[c])]
            if len(right_num) >= 2:
                end_col, begin_col = right_num[-2], right_num[-1]
            else:
                end_col, begin_col = df2.columns[-2], df2.columns[-1]
            left = [c for c in df2.columns if c not in (end_col, begin_col)]
            pick = left[:3] + [end_col, begin_col]
            df2 = df2[pick]
            df2.columns = ["CODE","NAME","NOTE","END","BEGIN"]
    else:
        df2 = pd.DataFrame(columns=["CODE","NAME","NOTE","END","BEGIN"])

    # 4) Nếu có bbox cell → OCR lại 2 cột số
    cell_boxes_all = []
    for item in table_items:
        res = item.get("res", {}) or {}
        raw_boxes = (res.get("boxes") or res.get("cell_boxes") or res.get("cells") or [])
        for cell in raw_boxes:
            r = cell.get("row", None) if isinstance(cell, dict) else None
            c = cell.get("col", None) if isinstance(cell, dict) else None
            bbox = cell.get("bbox", None) if isinstance(cell, dict) else None
            if bbox is None and isinstance(cell, dict):
                bbox = cell.get("box") or cell.get("bbox_xyxy") or cell.get("bbox_xywh")
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                x1,y1,x2,y2 = bbox
                cell_boxes_all.append((r, c, (int(x1),int(y1),int(x2),int(y2))))

    if cell_boxes_all:
        xs_all = np.array([ (bxy[0]+bxy[2]) / 2.0 for (_,_,bxy) in cell_boxes_all ], dtype=float)
        if xs_all.size >= 6:
            try:
                right_mask = xs_all > np.percentile(xs_all, 60)
                xs_right = xs_all[right_mask]
                if xs_right.size >= 4:
                    km = KMeans(n_clusters=2, n_init="auto", random_state=0).fit(xs_right.reshape(-1,1))
                    centers = sorted(km.cluster_centers_.ravel().tolist())
                    split_mid = float(sum(centers)/2.0)
                else:
                    split_mid = float(np.percentile(xs_all, 85))
            except Exception:
                split_mid = float(np.percentile(xs_all, 85))
        else:
            split_mid = float(np.percentile(xs_all, 85))

        df_fix = df2.copy()
        bgr_for_ocr = bgr  # dùng ảnh đã threshold/pad

        if not df_fix.empty:
            for (r,c,bxy) in cell_boxes_all:
                if r is None or c is None:
                    continue
                val = _ocr_crop_number(bgr_for_ocr, bxy, lang=OCR_LANG_DEFAULT)
                if r < len(df_fix) and val:
                    cx = (bxy[0] + bxy[2]) / 2.0
                    if cx < split_mid:
                        if "END" in df_fix.columns:
                            cur_end = str(df_fix.at[r, "END"])
                            if not re.fullmatch(r"\d{1,3}(?:\.\d{3})+|-?\d{4,}", cur_end or ""):
                                df_fix.at[r, "END"] = val
                    else:
                        if "BEGIN" in df_fix.columns:
                            cur_begin = str(df_fix.at[r, "BEGIN"])
                            if not re.fullmatch(r"\d{1,3}(?:\.\d{3})+|-?\d{4,}", cur_begin or ""):
                                df_fix.at[r, "BEGIN"] = val
            df2 = df_fix
        else:
            max_r = max([r for (r,_,_) in cell_boxes_all if r is not None] + [-1])
            rows = []
            for rr in range(max_r+1):
                rows.append(["","","","",""])
            for (r,c,bxy) in cell_boxes_all:
                if r is None: 
                    continue
                val = _ocr_crop_number(bgr_for_ocr, bxy, lang=OCR_LANG_DEFAULT)
                if val:
                    cx = (bxy[0] + bxy[2]) / 2.0
                    if cx < split_mid:
                        if not rows[r][3]:
                            rows[r][3] = val
                    else:
                        if not rows[r][4]:
                            rows[r][4] = val
            df2 = pd.DataFrame(rows, columns=["CODE","NAME","NOTE","END","BEGIN"])

    lines = []
    lines.append(" | ".join(df2.columns.tolist()))
    for _, r in df2.iterrows():
        vals = [str(x) if x is not None else "" for x in r.tolist()]
        if len(vals) >= 5:
            vals[3] = normalize_vn_amount(vals[3])
            vals[4] = normalize_vn_amount(vals[4])
        lines.append(" | ".join(vals))
    return "\n".join(lines)

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
def yaml_table_clean_rows(rows: List[Dict[str,str]], yaml_rules: Dict[str, Any]) -> List[Dict[str,str]]:
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
    import re
    expr = (expr or "").strip()
    if "=" not in expr: 
        return True, None, None
    left, right = [x.strip() for x in expr.split("=", 1)]

    def val(tok):
        tok = tok.strip()
        if tok.isdigit(): 
            return int(tok)
        key = tok.split(".")[0]
        return amap.get(key, 0)

    total, sign = 0, +1
    for tok in re.findall(r"[+-]|\d+(?:\.\d+)?", right):
        if tok in ["+","-"]:
            sign = +1 if tok == "+" else -1
        else:
            total += sign * val(tok)
    return (val(left) == total), val(left), total

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
    rows = assemble_financial_rows_from_pil(pil, y_tol=y_tol, lang=ocr_lang)
    pipe = table_rows_to_pipe(rows) if rows else ""
    rows_struct = parse_pipe_to_rows(pipe)
    metrics = score_table_quality(rows_struct)
    return pipe, metrics

def build_table_paddle(pil: Image.Image, paddle_lang: str, paddle_gpu: bool) -> Tuple[str, Dict[str,float]]:
    pipe = paddle_table_to_pipe(pil, lang=paddle_lang, use_gpu=paddle_gpu) or ""
    rows_struct = parse_pipe_to_rows(pipe)
    metrics = score_table_quality(rows_struct)
    return pipe, metrics

# ====== QUY TRÌNH 1 TRANG ======
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
    extra_blocks: có "TABLE→ROW-NARR" khi là TABLE và narrator_on=True
    """
    extra_blocks: List[Tuple[str, str]] = []
    gpt_used = False

    # OCR chọn engine: ưu tiên tesseract nếu ocr_engine="tesseract", ngược lại theo "auto"
    try:
        if ocr_engine == "paddle" or (ocr_engine == "auto" and _HAS_PADDLE and False):
            # (tắt ưu tiên paddle ở auto: chỉ dùng khi gọi riêng)
            ocr_txt, meta_json = ocr_image_text_paddle(pil, lang=paddle_lang, use_gpu=paddle_gpu)
        else:
            ocr_txt, meta_json = ocr_image_text(pil, lang=ocr_lang)
    except Exception as e:
        print(f"[WARN] OCR engine failed ({ocr_engine}): {e} → fallback Tesseract")
        ocr_txt, meta_json = ocr_image_text(pil, lang=ocr_lang)
    meta = json.loads(meta_json) if meta_json else {}
    meta["_table_format"] = table_format

    # ---- MIXED PAGE: phát hiện ROI bảng và tách TEXT / TABLE theo vùng ----
    extra_blocks_mixed: List[Tuple[str, str]] = []
    try:
        _bgr0 = cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)
        rois = _find_table_rois(_bgr0)
    except Exception:
        rois = []

    if rois and not force_table:
        # 1) OCR TEXT với vùng bảng đã che trắng
        try:
            pil_text_only = _mask_out_rois(pil, rois)
            txt_only, _meta_text = ocr_image_text(pil_text_only, lang=ocr_lang)
            block_text_text = apply_yaml_clean(txt_only, yaml_text, mode="text").strip()
        except Exception:
            block_text_text = ""

        # 2) Với mỗi ROI: build TABLE (TSV/KMeans ưu tiên; Paddle chỉ fallback)
        def _render_rows(rows_struct: List[Dict[str, str]]) -> str:
            if table_format == "pipe":
                return rows_to_pipe_min(rows_struct)
            if table_format == "json":
                return json.dumps(rows_struct, ensure_ascii=False, indent=2)
            return rows_to_ascii(rows_struct)

        meta.setdefault("crosschecks_roi", [])
        meta.setdefault("roi_table_metrics", [])
        meta.setdefault("gpt_used_roi", [])

        for (x1, y1, x2, y2) in rois:
            pad_top = int((yaml_table.get("globals") or {}).get("roi_pad_top", 40))
            pad_lr  = 4
            y1_p = max(0, int(y1) - pad_top)
            x1_p = max(0, int(x1) - pad_lr)
            x2_p = min(pil.width,  int(x2) + pad_lr)
            pil_roi = pil.crop((x1_p, y1_p, x2_p, int(y2)))
            meta.setdefault("roi_padded", []).append([x1_p, y1_p, x2_p, int(y2)])

            pipe_best, met_best, route_best = "", {"score": -1e9}, "none"
            if table_engine in ("auto", "tsv"):
                pipe_tsv, met_tsv = build_table_tsv(pil_roi, y_tol=y_tol, ocr_lang=ocr_lang)
                pipe_best, met_best, route_best = pipe_tsv, met_tsv, "tsv"

            need_pad = (table_engine == "paddle") or need_paddle_fallback(met_best)
            if _HAS_PADDLE and (table_engine in ("auto", "paddle")) and need_pad:
                try:
                    pipe_pad, met_pad = build_table_paddle(pil_roi, paddle_lang=paddle_lang, paddle_gpu=paddle_gpu)
                    if met_pad.get("score", -1e9) > met_best.get("score", -1e9):
                        pipe_best, met_best, route_best = pipe_pad, met_pad, "paddle"
                except Exception as e:
                    meta["paddle_table_error"] = str(e)

            pipe_best = _prefilter_table_lines(pipe_best, yaml_table)

            gpt_used_roi = False
            if (gpt_scope in ("table_only", "all")) and use_gpt and _HAS_GPT_ENHANCER and pipe_best.strip():
                try:
                    block2 = enhance_table_with_gpt(
                        table_text_cleaned=pipe_best,
                        image_pil=pil_roi,
                        meta={"company_hint": meta.get("company"), "period_hint": meta.get("period")},
                        mode=gpt_table_mode, model=gpt_model, temperature=gpt_temp, log_diag=log_gpt,
                    )
                    if isinstance(block2, str) and block2.strip():
                        pipe_best = block2.strip()
                        gpt_used_roi = True
                except Exception as e:
                    meta.setdefault("gpt_errors_roi", []).append(str(e))
            meta["gpt_used_roi"].append(bool(gpt_used_roi))

            rows_struct = parse_pipe_to_rows(pipe_best)
            rows_struct = yaml_table_clean_rows(rows_struct, yaml_table.get("globals", {}))
            if do_autofix:
                rows_struct, report = validate_and_autofix_rows(rows_struct)
            else:
                report = {"metrics": score_table_quality(rows_struct), "issues": []}

            roi_metrics = score_table_quality(rows_struct)
            meta["roi_table_metrics"].append({
                "roi": [x1_p, y1_p, x2_p, int(y2)],
                "route": route_best,
                "metrics": roi_metrics
            })

            cross_ok, cross_detail = None, None
            try:
                cross_ok, cross_detail = run_crosschecks(rows_struct, yaml_table)
                meta["crosschecks_roi"].append({"ok": cross_ok, "detail": cross_detail})
            except Exception as e:
                meta["crosschecks_roi"].append({"ok": None, "error": str(e)})

            tbl_text = _render_rows(rows_struct)
            extra_blocks_mixed.append(("TABLE", tbl_text))

            if narrator_on:
                try:
                    row_narr = narrator_rows(rows_struct)
                except Exception as e:
                    row_narr = f"[Narrative lỗi: {e}]"
                extra_blocks_mixed.append(("TABLE→ROW-NARR", row_narr))

                if isinstance(cross_detail, list):
                    lines = ["[Cross-check] " + ("OK ✅" if cross_ok else "FAIL ❌")]
                    for r in cross_detail:
                        e, s = r["end"], r["start"]
                        e_msg = "OK" if e["ok"] else f"FAIL ({e['lhs']} ≠ {e['rhs']})"
                        s_msg = "OK" if s["ok"] else f"FAIL ({s['lhs']} ≠ {s['rhs']})"
                        lines.append(f"- {r['name']}: END {e_msg}; BEGIN {s_msg}")
                    extra_blocks_mixed.append(("TABLE→CROSS", "\n".join(lines)))

        if extra_blocks_mixed:
            meta.update({
                "mixed_page": True,
                "roi_count": len(rois),
                "engine": meta.get("engine"),
            })
            return "TEXT", block_text_text, meta, extra_blocks_mixed

    # ---- Quyết định TABLE cho cả trang? ----
    try:
        is_tbl = is_table_page(ocr_txt, yaml_table) or force_table
    except TypeError:
        is_tbl = is_table_page(ocr_txt, {}) or force_table

    if is_tbl:
        best_pipe, best_metrics, best_route = "", {"score": -1e9}, "none"

        if table_engine in ("auto", "tsv"):
            pipe_tsv, met_tsv = build_table_tsv(pil, y_tol=y_tol, ocr_lang=ocr_lang)
            best_pipe, best_metrics, best_route = pipe_tsv, met_tsv, "tsv"

        need_pad = (table_engine == "paddle") or need_paddle_fallback(best_metrics)
        if _HAS_PADDLE and (table_engine in ("auto", "paddle")) and need_pad:
            try:
                pipe_pad, met_pad = build_table_paddle(pil, paddle_lang=paddle_lang, paddle_gpu=paddle_gpu)
                if met_pad.get("score", -1e9) > best_metrics.get("score", -1e9):
                    best_pipe, best_metrics, best_route = pipe_pad, met_pad, "paddle"
            except Exception as e:
                meta["paddle_table_error"] = str(e)

        if not best_pipe:
            cleaned = apply_yaml_clean(ocr_txt, yaml_table, mode="table")
            maybe_pipe = cleaned if ("|" in cleaned) else coerce_pipe_table(cleaned)
            best_pipe = maybe_pipe or ""
            best_metrics = score_table_quality(parse_pipe_to_rows(best_pipe))
            best_route = best_route if best_route != "none" else "coerce"

        best_pipe = _prefilter_table_lines(best_pipe, yaml_table)

        if (gpt_scope in ("table_only", "all")) and use_gpt and _HAS_GPT_ENHANCER and best_pipe.strip():
            try:
                block2 = enhance_table_with_gpt(
                    table_text_cleaned=best_pipe,
                    image_pil=pil,
                    meta={"company_hint": meta.get("company"), "period_hint": meta.get("period")},
                    mode=gpt_table_mode, model=gpt_model, temperature=gpt_temp, log_diag=log_gpt,
                )
                if isinstance(block2, str) and block2.strip():
                    best_pipe = block2.strip()
                    gpt_used = True
            except Exception as e:
                meta["gpt_error"] = str(e)

        rows = parse_pipe_to_rows(best_pipe)
        rows = yaml_table_clean_rows(rows, yaml_table.get("globals", {}))
        if do_autofix:
            rows, report = validate_and_autofix_rows(rows)
        else:
            report = {"metrics": score_table_quality(rows), "issues": []}

        try:
            cross_ok, cross_detail = run_crosschecks(rows, yaml_table)
            meta["crosscheck"] = {"ok": cross_ok, "detail": cross_detail}
        except Exception as e:
            meta["crosscheck"] = {"ok": None, "error": str(e)}

        fmt = (table_format or "pipe").lower()
        if fmt == "ascii":
            block_text = rows_to_ascii(rows)
        elif fmt == "json":
            block_text = rows_to_json_min(rows)
        else:
            block_text = rows_to_pipe_min(rows)

        extra_blocks = []
        if narrator_on:
            try:
                row_narr = narrator_rows(rows)
            except Exception as e:
                row_narr = f"[Narrative lỗi: {e}]"
            extra_blocks = [("TABLE→ROW-NARR", row_narr)]

            if isinstance(meta.get("crosscheck", {}).get("detail"), list):
                lines = ["[Cross-check] " + ("OK ✅" if meta["crosscheck"]["ok"] else "FAIL ❌")]
                for r in meta["crosscheck"]["detail"]:
                    e, s = r["end"], r["start"]
                    e_msg = "OK" if e["ok"] else f"FAIL ({e['lhs']} ≠ {e['rhs']})"
                    s_msg = "OK" if s["ok"] else f"FAIL ({s['lhs']} ≠ {s['rhs']})"
                    lines.append(f"- {r['name']}: END {e_msg}; BEGIN {s_msg}")
                extra_blocks.append(("TABLE→CROSS", "\n".join(lines)))

        meta.update({
            "gpt_used": gpt_used,
            "table_route": best_route,
            "table_metrics": best_metrics,
            "validator_report": report,
            "text_sha1": _sha1(block_text),
            "engine": meta.get("engine"),
            "table_format": fmt,
        })
        return "TABLE", block_text, meta, extra_blocks

    # === TEXT nhánh ===
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
    # nạp YAML
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
    exts = (".pdf", ".docx", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
    files = [
        p for p in glob.glob(os.path.join(args.input_root, "**", "*"), recursive=True)
        if os.path.isfile(p) and os.path.splitext(p)[1].lower() in exts
    ]
    print(f"[DEBUG] input-root = {args.input_root}")
    print(f"[DEBUG] found {len(files)} file(s)")
    if not files:
        print(f"⚠️ Không tìm thấy file hợp lệ dưới: {args.input_root}")
        return
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
        except Exception as e:
            import traceback
            print(f"❌ Lỗi file: {fp} → {e}")
            traceback.print_exc()
    print("\n✅ Hoàn tất. Mỗi file input sinh ra 1 TXT (TEXT/TABLE + diễn giải dòng) + 1 meta.json.")
    print("   Bật --split-debug để có thêm file _TEXT/_TABLE; dùng --narrator n để tắt diễn giải khi QA.")

if __name__ == "__main__":
    main()
