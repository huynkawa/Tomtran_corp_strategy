# -*- coding: utf-8 -*-
"""
P1A (GPT) — Hybrid Auto-Route (OpenCV+TSV → Paddle) [FULL/Hardened, VAR-COLS + PRELIGHT]
- KHÔNG ép 5 cột; dựng bảng theo số cột thực tế (var-col)
- Auto score/route engine cho TSV/Paddle (ưu tiên đầu ra “đầy” dòng)
- Lọc header/caption “rác bảng”; reflow TEXT; OCR Tesseract/Paddle
- TÍCH HỢP PRELIGHT (deskew + crop margins + adaptive binarize nhẹ)
- Phát hiện MIXED mạnh hơn (mask top/bottom + hạ ngưỡng diện tích ROI)
- Xuất JSONL vector store (_vector.jsonl), mỗi dòng = 1 chunk (content + metadata)

I/O:
- Mỗi file input → 1 TXT (### [PAGE XX] [TEXT]/[TABLE]/[TABLE→ROI] ...) + 1 META.json + 1 VECTOR.jsonl
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

#========BẬT TẮT CÔNG TẮC=======
# ==== MASTER SWITCH CHO GPT ====
# None  = theo CLI (mặc định hiện tại)
# True  = ép BẬT mọi bước GPT
# False = ép TẮT mọi bước GPT
GPT_MASTER_SWITCH = True

# ---- RAW toggle (bỏ qua YAML & prefilter) ----
P1A_RAW_MODE = False   # True = chạy thô, False = chạy theo YAML


# [REPLACE] ==== GPT helper: chỉ xử lý TABLE (var-cols pipe) ====
def _maybe_gpt_enhance_table(
    pipe_text: str,
    pil_img,                       # ROI hoặc cả trang (PIL.Image.Image)
    *,
    gpt_table_enabled: bool,
    gpt_scope: str,
    gpt_table_mode: str,
    gpt_model: str,
    gpt_temp: float,
    log_gpt: bool
):
    """
    Trả về: (pipe_out, used: bool, err: Optional[str])
    - Chỉ chạy khi gpt_table_enabled=True và scope cho phép ("table_only" hoặc "all").
    - Giữ định dạng PIPE (var-cols). Fallback pipe_text nếu lỗi.
    """
    if not (pipe_text and pipe_text.strip()):
        return pipe_text, False, None
    if not gpt_table_enabled or gpt_scope not in ("table_only", "all"):
        return pipe_text, False, None

    # Tôn trọng trạng thái module enhancer
    if not globals().get("_HAS_GPT_ENHANCER", False):
        if log_gpt:
            print("   [GPT TABLE][SKIP] Module enhancer chưa sẵn sàng (_HAS_GPT_ENHANCER=False).")
        return pipe_text, False, None

    # Kiểm tra ảnh hợp lệ
    if not isinstance(pil_img, Image.Image):
        if log_gpt:
            print("   [GPT TABLE][SKIP] pil_img không phải PIL.Image → bỏ qua.")
        return pipe_text, False, None

    try:
        # Dùng hàm enhancer đã import (nếu có)
        out = enhance_table_with_gpt(
            pipe_text,
            pil_img,
            meta=None,
            mode=gpt_table_mode,       # "financial" | "generic" | "auto"
            model=gpt_model,
            temperature=gpt_temp,
            log_diag=log_gpt
        )
        out = (out or pipe_text).strip()
        return out, (out != pipe_text), None
    except Exception as e:
        return pipe_text, False, str(e)


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
OUTPUT_ROOT_DEFAULT = r"D:\1.TLAT\3. ChatBot_project\1_Insurance_Strategy\outputs\p1a_clean10_ocr_bctc_GPT_version3"
YAML_TABLE_DEFAULT  = r"D:\1.TLAT\3. ChatBot_project\1_Insurance_Strategy\configs\p1a_clean10_ocr_bctc_table.yaml"
YAML_TEXT_DEFAULT   = r"D:\1.TLAT\3. ChatBot_project\1_Insurance_Strategy\configs\p1a_clean10_ocr_bctc_text.yaml"


# ====== PRELIGHT (deskew + crop + adaptive binarize) ======
def estimate_skew_angle(gray: np.ndarray) -> float:
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=160,
                            minLineLength=max(gray.shape)//10, maxLineGap=20)
    if lines is None:
        return 0.0
    angles = []
    for l in lines:
        x1, y1, x2, y2 = l[0]
        ang = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        while ang >= 90: ang -= 180
        while ang <  -90: ang += 180
        angles.append(ang)
    if not angles:
        return 0.0
    a = np.array(angles)
    near_h = a[np.abs(a) <= 30]
    near_v = a[(np.abs(np.abs(a) - 90) <= 30)]
    if len(near_h) >= len(near_v):
        return float(np.median(near_h)) if len(near_h) else 0.0
    deltas = []
    for v in near_v:
        deltas.append(v - 90 if v > 0 else v + 90)
    return float(np.median(deltas)) if deltas else 0.0

def rotate_bound(img: np.ndarray, angle_deg: float) -> np.ndarray:
    (h, w) = img.shape[:2]
    c = (w//2, h//2)
    M = cv2.getRotationMatrix2D(c, angle_deg, 1.0)
    cos, sin = abs(M[0,0]), abs(M[0,1])
    nW = int((h*sin) + (w*cos)); nH = int((h*cos) + (w*sin))
    M[0,2] += (nW/2) - c[0]; M[1,2] += (nH/2) - c[1]
    return cv2.warpAffine(img, M, (nW, nH), flags=cv2.INTER_CUBIC, borderValue=(255,255,255))

def crop_margins(gray: np.ndarray, pad: int=16) -> Tuple[np.ndarray, Tuple[int,int,int,int]]:
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    inv = 255 - thr
    ys, xs = np.where(inv > 0)
    if len(xs) == 0 or len(ys) == 0:
        return gray, (0,0,gray.shape[1], gray.shape[0])
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
    x1 = min(gray.shape[1]-1, x1 + pad)
    y1 = min(gray.shape[0]-1, y1 + pad)
    return gray[y0:y1+1, x0:x1+1], (x0,y0,x1-x0+1,y1-y0+1)

def gentle_binarize(gray: np.ndarray) -> np.ndarray:
    blur = cv2.medianBlur(gray, 3)
    return cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_,
                                 cv2.THRESH_BINARY, 31, 15)

def _prelight_apply(pil: Image.Image,
                    mode: str = "auto",
                    choose_layer: str = "auto",
                    log: bool = False) -> Tuple[Image.Image, Dict[str, Any], Optional[Image.Image]]:
    """
    mode: "auto" | "on" | "off"
    choose_layer: "auto" | "bin" | "gray"
    Trả về: (PIL dùng cho OCR/ROI, meta, PIL_bin_nếu_có)
    """
    bgr0 = cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    gray0 = cv2.cvtColor(bgr0, cv2.COLOR_BGR2GRAY)
    H, W = gray0.shape[:2]
    angle = estimate_skew_angle(gray0)

    used = False
    bgr = bgr0.copy()
    gray = gray0

    # quyết định bật prelight theo mode
    use_prelight = False
    if mode == "on":
        use_prelight = True
    elif mode == "off":
        use_prelight = False
    else:
        # auto: bật nếu nghiêng đáng kể hoặc nền không đủ trắng
        med = float(np.median(gray0))
        use_prelight = (abs(angle) > 0.3) or (med < 245.0)

    crop_box = (0,0,W,H)
    bin_img = None

    if use_prelight:
        used = True
        if abs(angle) > 0.3:
            bgr = rotate_bound(bgr, -angle)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray_c, crop_box = crop_margins(gray, pad=16)
        gray = gray_c
        bin_img = gentle_binarize(gray)

    # chọn lớp cho OCR
    if choose_layer == "bin":
        layer = "bin"
    elif choose_layer == "gray":
        layer = "gray"
    else:
        # auto: nếu nền tối (median<245) hoặc chữ mảnh → dùng bin
        med = float(np.median(gray))
        layer = "bin" if (med < 245.0 and bin_img is not None) else "gray"

    if layer == "bin" and bin_img is not None:
        out_bgr = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)
        pil_out = Image.fromarray(cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB))
    else:
        out_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        pil_out = Image.fromarray(cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB))

    meta = {
        "prelight_used": used,
        "prelight_mode": mode,
        "prelight_layer": layer,
        "skew_angle_deg": round(float(angle), 3),
        "crop_box_xywh": crop_box,
        "median_gray": float(np.median(gray0)),
    }
    if log:
        print(f"   [PRELIGHT] used={used} | layer={layer} | angle={meta['skew_angle_deg']}° | crop={crop_box}")

    pil_bin = Image.fromarray(bin_img) if bin_img is not None else None
    return pil_out, meta, pil_bin


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

# === Vector-store helpers ===
def _normalize_for_vector(text: str) -> str:
    """Rút gọn khoảng trắng, bỏ ký tự rác, giữ cấu trúc cơ bản để embed."""
    s = (text or "")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"[ \t]*\n[ \t]*", "\n", s)
    drop_lines = []
    for ln in s.splitlines():
        low = ln.lower()
        if re.search(r"\b(vnd|vnđ|đơn vị|don vi|as at|as of|for the year ended)\b", low):
            continue
        if len(ln) > 280 and not re.search(r"[0-9]", ln):
            continue
        drop_lines.append(ln.strip())
    s = "\n".join(drop_lines).strip()
    return s

def _make_chunk(content: str, meta: dict) -> Optional[dict]:
    c = (content or "").strip()
    if not c:
        return None
    return {
        "content": _normalize_for_vector(c),
        "metadata": meta
    }

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

# ====== HEADER/HEURISTICS ======
_VN_TOK = lambda s: _strip_vn_accents((s or "")).lower()

def _looks_like_5col_header(words: List[str]) -> bool:
    """
    Trả về True nếu 1 hàng có đủ tín hiệu của header 5 cột:
    'ma so' | 'tai san' | 'thuyet minh' | 'so cuoi nam' | 'so dau nam'
    Chấp nhận sai chính tả nhẹ.
    """
    ws = [_VN_TOK(w) for w in words if w and isinstance(w, str)]
    joined = " ".join(ws)
    want = [
        ("ma so", "ma  so", "ma sõ", "ma s0"),
        ("tai san", "tai san"),
        ("thuyet minh", "thuyet  minh", "thuyet  minh"),
        ("so cuoi nam", "so cuo i nam", "so cuoi  nam"),
        ("so dau nam", "so dau  nam")
    ]
    hit = 0
    for alts in want:
        if any(a in joined for a in alts):
            hit += 1
    return hit >= 3  # nới lỏng: 3/5 token là coi như header 5 cột

_NUM_RE = re.compile(r"^-?\(?\d{1,3}(?:[.,]\d{3})+(?:\)?)*$")
def _is_num_cell(s: str) -> bool:
    s = (s or "").strip()
    if not s: return False
    s = s.replace(" ", "")
    return bool(_NUM_RE.match(s)) or bool(re.fullmatch(r"-?\d{4,}", s))


def _merge_adjacent_text_columns(cells: List[str], want_k: Optional[int]) -> List[str]:
    """
    Ghép 2 cột text liền nhau nếu:
      - cả hai đều không phải số
      - bên phải còn ít nhất 2 cột số lớn (cuối năm/đầu năm)
    Mục tiêu: sửa các tách kiểu 'Tiền và các | khoản tương đương'.
    """
    if not cells: return cells[:]
    c = cells[:]
    # tìm 2 cột số ở rìa phải
    right_numeric = [i for i,x in enumerate(c) if _is_num_cell(x)]
    if len(right_numeric) >= 2:
        last2 = right_numeric[-2:]
        cutoff = min(last2)  # mọi ghép chỉ xét trước 2 cột số cuối
        i = 1
        while i < cutoff:
            if not _is_num_cell(c[i-1]) and not _is_num_cell(c[i]):
                c[i-1] = (c[i-1] + " " + c[i]).strip()
                del c[i]
                cutoff -= 1
                if want_k and len(c) <= want_k: break
                continue
            i += 1
    return c


# ==== AUTO COLUMN INFERENCE (always-on) =====================================

_RE_CODE = re.compile(r"^\d{3}(?:\.\d+)?$")                # 100, 151.1 ...
_RE_NOTE = re.compile(r"^\d+(?:\([a-z]\))?$", re.I)        # 4  | 5.2 | 15.1 | 7(a)
_RE_AMT1 = re.compile(r"^-?\(?\d{1,3}(?:[.,]\d{3})+(?:\)?)*$")  # 1.234.567 | (2.345)
_RE_AMT2 = re.compile(r"^-?\d{4,}$")                       # 12345
# Giá trị placeholder cho ô rỗng để không bị “co” cột khi đọc
NULL_TOKEN = "None"

def _is_amount_cell(s: str) -> bool:
    s = (s or "").strip().replace(" ", "")
    return bool(_RE_AMT1.fullmatch(s)) or bool(_RE_AMT2.fullmatch(s))

_AMT_GRP = re.compile(r"(-?\(?\d{1,3}(?:[.,]\d{3})+(?:\)?)*|-?\d{4,})")

def _split_two_amounts_in_cell(cells: List[str]) -> List[str]:
    """
    Nếu ô bên phải đang chứa đồng thời 2 số tiền (END và BEGIN) dính nhau,
    ví dụ: '15.057.409.157 55.847.772.902' → tách thành 2 ô riêng.
    Ưu tiên tách ở cell phải nhất chưa phải là cột số chuẩn.
    """
    if not cells:
        return cells

    # duyệt từ phải sang trái, tìm cell có >=2 nhóm 'số tiền'
    for j in range(len(cells) - 1, -1, -1):
        raw = (cells[j] or "").strip()
        if not raw or "|" in raw:
            continue
        hits = _AMT_GRP.findall(raw.replace(",", "."))
        if len(hits) >= 2:
            # lấy 2 số cuối làm (END, BEGIN)
            endv = normalize_vn_amount(hits[-2])
            begv = normalize_vn_amount(hits[-1])
            # phần text còn lại (nếu có) bỏ đi để tránh nhiễu
            cells = cells[:j] + [endv, begv] + cells[j+1:]
            break
    return cells


def _is_code_cell(s: str) -> bool:
    return bool(_RE_CODE.fullmatch((s or "").strip()))

def _is_note_cell(s: str) -> bool:
    return bool(_RE_NOTE.fullmatch((s or "").strip()))

def _split_rows(pipe_text: str):
    lines = [ln for ln in (pipe_text or "").splitlines() if ln.strip()]
    if not lines: return [], []
    header = [c.strip() for c in lines[0].split("|")]
    rows = [[c.strip() for c in ln.split("|")] for ln in lines[1:]]
    return header, rows

def _infer_schema_indices_from_rows(rows: List[List[str]], look_rows: int = 30) -> Dict[str, Any]:
    """
    Suy luận vị trí các cột: code/name/note + danh sách cột tiền (phải).
    Không dựa vào header, chỉ dựa vào 'hình dạng' nhiều dòng.
    """
    if not rows:
        return {"code": None, "note": None, "name": None, "amount_cols": []}

    # Giới hạn số dòng quan sát để giảm nhiễu
    sample = rows[:max(5, min(look_rows, len(rows)))]
    nmax = max(len(r) for r in sample)

    # Pad để dễ duyệt
    pad_rows = [ (r + [""]*(nmax - len(r))) for r in sample ]

    stats = []
    for j in range(nmax):
        col_vals = [r[j] for r in pad_rows]
        numeric = sum(1 for x in col_vals if _is_amount_cell(x))
        codehit = sum(1 for x in col_vals if _is_code_cell(x))
        notehit = sum(1 for x in col_vals if _is_note_cell(x))
        avglen  = sum(len(x) for x in col_vals) / max(1, len(col_vals))
        stats.append({
            "j": j, "numeric": numeric, "codehit": codehit, "notehit": notehit, "avglen": avglen
        })

    # 1) Cột tiền: chọn tất cả cột có 'numeric' cao, ưu tiên ở BÊN PHẢI
    #    Ngưỡng: xuất hiện số ở >= 35% số dòng mẫu
    thr_num = max(2, int(0.35*len(pad_rows)))
    amount_cols = [s["j"] for s in stats if s["numeric"] >= thr_num]
    amount_cols.sort()
    # đảm bảo "bên phải": nếu rải rác, lấy cụm phải nhất
    if len(amount_cols) >= 3:
        # giữ cụm phải với >=2 cột
        right_pairs = [amount_cols[i:] for i in range(len(amount_cols)) if len(amount_cols[i:]) >= 2]
        if right_pairs:
            amount_cols = right_pairs[-1]

    # 2) CODE: cột có codehit cao, ưu tiên bên trái
    code_col = None
    cand_code = [s for s in stats if s["codehit"] >= max(2, int(0.25*len(pad_rows)))]
    if cand_code:
        cand_code.sort(key=lambda x: (x["j"], -x["codehit"]))
        code_col = cand_code[0]["j"]
    else:
        # fallback: nếu cột 0 có nhiều dòng trông giống code → chọn 0
        if stats and stats[0]["codehit"] >= 1:
            code_col = 0

    # 3) NOTE: cột nhỏ xen giữa name và tiền
    note_col = None
    cand_note = [s for s in stats if s["notehit"] >= 1]
    if cand_note:
        # Ưu tiên cột có avglen nhỏ, ở giữa code và cột tiền đầu tiên
        amt_min = amount_cols[0] if amount_cols else nmax+2
        best = sorted(cand_note, key=lambda x: (x["avglen"], abs(x["j"] - min(amt_min, nmax))))[0]
        note_col = best["j"]

    # 4) NAME: cột text dài nhất ở vùng trái (không đụng code/note)
    name_col = None
    left_zone = [s for s in stats if s["j"] < (amount_cols[0] if amount_cols else nmax)]
    left_zone.sort(key=lambda x: (-x["avglen"], x["j"]))
    for s in left_zone:
        if s["j"] != code_col and s["j"] != note_col and s["numeric"] < thr_num:
            name_col = s["j"]; break

    return {"code": code_col, "note": note_col, "name": name_col, "amount_cols": amount_cols}

def _project_row_to_flex_schema(cells: List[str], idx: Dict[str, Any]) -> Dict[str, str]:
    """Dựng record linh hoạt: CODE/NAME/NOTE + END/BEGIN (nếu có) + EXTRA_VALUES."""
    k = len(cells)
    code  = cells[idx["code"]]  if idx.get("code")  is not None and idx["code"]  < k else ""
    name  = cells[idx["name"]]  if idx.get("name")  is not None and idx["name"]  < k else ""
    note  = cells[idx["note"]]  if idx.get("note")  is not None and idx["note"]  < k else ""

    # Chuẩn hoá số tiền và lấy 2 cột phải nhất làm END/BEGIN nếu có >=2
    amt_cols = [c for c in (idx.get("amount_cols") or []) if c < k]
    amts = [normalize_vn_amount(cells[j]) for j in amt_cols]
    endv, begv = "", ""
    if len(amts) >= 2:
        endv, begv = amts[-2], amts[-1]
    elif len(amts) == 1:
        endv = amts[0]

    # Extra values (khi >2 cột số)
    extras = amts[:-2] if len(amts) > 2 else []

    return {
        "CODE": code, "NAME": name, "NOTE": note,
        "END": endv, "BEGIN": begv,
        "EXTRA_VALUES": extras
    }

def pipe_to_flex_schema(pipe_text: str) -> Dict[str, Any]:
    """
    Chuyển pipe var-cols → {header, rows[], inferred_indices}
    rows[i] theo schema linh hoạt ở trên.
    """
    header, rows = _split_rows(pipe_text)
    if not rows:
        return {"header": header, "rows": [], "indices": {"code":None,"note":None,"name":None,"amount_cols":[]}}

    # ghép text bị tách đôi trước 2 cột số (giống logic đã có)
    # (chỉ thực hiện tại runtime theo từng dòng)
    maxk = max(len(r) for r in rows)
    fused = []
    for r in rows:
        r2 = _merge_adjacent_text_columns(r[:], want_k=None)
        fused.append(r2)

    idx = _infer_schema_indices_from_rows(fused)

    # dự án từng dòng thành record
    out_rows = []
    for cells in fused:
        rec = _project_row_to_flex_schema(cells, idx)
        # nếu trống hết => bỏ
        if any([rec["CODE"], rec["NAME"], rec["END"], rec["BEGIN"]]):
            out_rows.append(rec)

    return {"header": header, "rows": out_rows, "indices": idx}


def _repack_pipe_by_inferred_schema(pipe_text: str,
                                    *,
                                    fill_note_with_none: bool = True,
                                    force_five_when_fin_like: bool = True) -> str:
    """
    Dựng lại bảng 5 cột chuẩn tài chính (CODE | NAME | Thuyết minh | Số cuối năm | Số đầu năm)
    dựa trên auto-schema đã suy luận. Chỉ điền 'None' cho cột NOTE (nếu trống).
    Các cột tiền nếu trống thì để rỗng "" (không ghi None).

    - Không mang theo các cột text dư ở giữa; chỉ dùng đúng CODE/NAME/NOTE + END/BEGIN
    - Không lặp lại CODE|NAME lần 2.
    """
    if not pipe_text or not pipe_text.strip():
        return pipe_text

    model = pipe_to_flex_schema(pipe_text)
    rows  = model.get("rows") or []

    # Nếu bảng không có "dáng" VAS/financial thì trả nguyên
    s0 = (pipe_text.splitlines() or [""])[0].lower()
    looks_fin = ("mã số" in s0 or "ma so" in s0) or ("thuyết minh" in s0 or "thuyet minh" in s0) \
                or ("số cuối năm" in s0 or "so cuoi nam" in s0) or ("số đầu năm" in s0 or "so dau nam" in s0)
    if force_five_when_fin_like and not looks_fin and not rows:
        return pipe_text

    header = ["CODE", "NAME", "Thuyết minh", "Số cuối năm", "Số đầu năm"]
    out = [" | ".join(header)]

    for r in rows:
        code  = (r.get("CODE") or "").strip()
        name  = (r.get("NAME") or "").strip()
        note  = (r.get("NOTE") or "").strip()
        endv  = (r.get("END")  or "").strip()
        begv  = (r.get("BEGIN") or "").strip()

        # Chỉ NOTE mới được điền 'None' khi trống
        if not note and fill_note_with_none:
            note = "None"

        # Không cho cột tiền thành 'None'
        if endv.lower() == "none":  endv = ""
        if begv.lower() == "none":  begv = ""

        out.append(" | ".join([code, name, note, endv, begv]))

    # Nếu không thu được dòng nào từ auto-schema → trả nguyên
    return "\n".join(out) if len(out) > 1 else pipe_text




def _repack_pipe_by_inferred_schema(pipe_text: str) -> str:
    """
    Dựa trên pipe_to_flex_schema:
    - Nếu nhận diện được bảng tài chính (có CODE và >=1 cột tiền), rebuild về:
        CODE | NAME | NOTE? | END | BEGIN
      (NOTE chỉ có khi ít nhất một dòng có NOTE khác rỗng)
    - Nếu KHÔNG nhận diện được (bảng generic) → giữ nguyên pipe gốc.
    - Loại bỏ hoàn toàn các dòng rỗng hoặc chỉ có '|' / '\' rác.
    """
    model = pipe_to_flex_schema(pipe_text or "")
    rows = model.get("rows") or []
    idx  = model.get("indices") or {}

    # Không phải bảng tài chính → trả nguyên
    has_code = idx.get("code") is not None
    amt_cols = idx.get("amount_cols") or []
    if not rows or not has_code or not amt_cols:
        # dọn rác nhẹ: bỏ dòng chỉ chứa dấu | hoặc '\' 
        clean_lines = []
        for ln in (pipe_text or "").splitlines():
            t = (ln or "").strip()
            if not t or re.fullmatch(r"[|\\\s]+", t):
                continue
            clean_lines.append(ln)
        return "\n".join(clean_lines)

    need_note = any((r.get("NOTE") or "").strip() for r in rows)
    header = ["CODE", "NAME"]
    if need_note: header.append("Thuyết minh")
    header += ["Số cuối năm", "Số đầu năm"]

    out_lines = [" | ".join(header)]
    for r in rows:
        code  = (r.get("CODE") or "").strip() or NULL_TOKEN
        name  = (r.get("NAME") or "").strip() or NULL_TOKEN
        note  = (r.get("NOTE") or "").strip() or NULL_TOKEN
        endv  = (r.get("END")  or "").strip() or NULL_TOKEN
        begv  = (r.get("BEGIN")or "").strip() or NULL_TOKEN

        if need_note:
            cells = [code, name, note, endv, begv]
        else:
            cells = [code, name, endv, begv]
        out_lines.append(" | ".join(cells))

    return "\n".join(out_lines)




def _pad_to_k(cells: List[str], k: int) -> List[str]:
    """Bảo đảm đúng k cột; mọi ô trống → NULL_TOKEN."""
    out = (cells or [])[:k] + ["" for _ in range(max(0, k - len(cells or [])))]
    return [ (c if (c is not None and str(c).strip() != "") else NULL_TOKEN) for c in out ]


def _auto_insert_virtual_note_col(pipe_text: str) -> str:
    """
    Dựa trên suy luận cột (pipe_to_flex_schema):
    - Nếu có CODE + NAME + ≥1 amount_col và KHÔNG tìm ra cột NOTE riêng,
      chèn 1 cột 'NOTE' rỗng ngay trước cột tiền đầu tiên.
    - Không đụng vào bảng generic (không có CODE) → giữ nguyên.
    """
    if not pipe_text.strip():
        return pipe_text
    header = (pipe_text.splitlines() or [""])[0]
    body   = (pipe_text.splitlines() or [])[1:]
    if not body:
        return pipe_text

    model = pipe_to_flex_schema(pipe_text)
    idx   = model.get("indices") or {}
    code_col = idx.get("code")
    name_col = idx.get("name")
    note_col = idx.get("note")
    amt_cols = idx.get("amount_cols") or []

    # điều kiện chèn NOTE ảo
    if code_col is None or name_col is None or note_col is not None or not amt_cols:
        return pipe_text

    insert_at = min(amt_cols)  # vị trí cột tiền đầu tiên
    new_lines = []

    # header: đổi nhãn cho đẹp (tuỳ ý)
    h_cells = [c.strip() for c in header.split("|")]
    if len(h_cells) < insert_at+1:
        # header pipe dạng C1|C2... -> cứ chèn thêm 1 nhãn
        h_cells = h_cells + [""] * (insert_at+1-len(h_cells))
    h_cells.insert(insert_at, "Thuyết minh")
    new_lines.append(" | ".join(h_cells))

    # body
    for ln in body:
        cells = [c.strip() for c in ln.split("|")]
        if len(cells) < insert_at:
            cells = cells + [""] * (insert_at-len(cells))
        cells.insert(insert_at, "")   # chèn NOTE rỗng
        new_lines.append(" | ".join(cells))

    return "\n".join(new_lines)

# ==== hàm “vá” hàng bị tách đôi text & chuẩn hóa số cột

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
    if re.search(r"Ý kiến của Ki(ê|e)m toán|Ki(ê|e)m toán viên|Auditor'?s opinion", text, re.I):
        return False
    det = (yaml_table.get("globals") or {}).get("detection_tokens") or {}
    must_any = det.get("must_have_any") or []
    hit_tokens = any(re.search(re.escape(tok), text, re.I) for tok in must_any)
    code_hits   = len(CODE_LINE.findall(text))
    money_lines = sum(1 for ln in text.splitlines() if len(MONEY.findall(ln)) >= 2)
    dense_struct = (code_hits >= 4 and money_lines >= 4)
    return bool(hit_tokens or dense_struct)

def is_table_page(txt: str, yaml_table: dict) -> bool:
    return _looks_like_table(txt, yaml_table)

# ====== TSV→Hàng/Cột (no-line) ======
def _norm_word(w: str) -> str:
    return re.sub(r"\s+", " ", (w or "").strip())

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

# ====== TABLE ROI DETECTOR (morphology) ======
def _find_table_rois(bgr: np.ndarray, min_area_ratio: float = 0.03) -> List[Tuple[int,int,int,int]]:
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
    area_min = (w*h) * float(min_area_ratio)  # CHỈNH: hạ ngưỡng còn 3% mặc định
    for c in cnts:
        x,y,ww,hh = cv2.boundingRect(c)
        area = ww*hh
        if area < area_min:
            continue
        if y < int(0.07*h):  # tránh ăn lên header
            continue
        if hh < h*0.08 and ww > w*0.6:  # loại band ngang mỏng
            continue
        rois.append((x, y, x+ww, y+hh))
    rois.sort(key=lambda b: b[1])
    return rois

def _mask_out_rois(pil: Image.Image, rois: List[Tuple[int,int,int,int]]) -> Image.Image:
    bgr = cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    for (x1,y1,x2,y2) in rois:
        cv2.rectangle(bgr, (x1,y1), (x2,y2), (255,255,255), thickness=-1)
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

# ---- NEW: không mask phần header khi che ROI để OCR TEXT ----
def _guard_header_for_text_mask(rois: List[Tuple[int,int,int,int]], page_h: int,
                                guard_ratio: float = 0.12) -> List[Tuple[int,int,int,int]]:
    """
    Không mask vùng trên cùng (header) ~ guard_ratio chiều cao trang để không mất TEXT.
    """
    if not rois:
        return rois
    guard = int(max(24, page_h * guard_ratio))
    out = []
    for (x1,y1,x2,y2) in rois:
        y1_adj = max(y1, guard)   # đẩy mép trên ROI xuống dưới vùng header
        if y2 - y1_adj > 12:      # vẫn còn đủ cao mới giữ
            out.append((x1, y1_adj, x2, y2))
    return out

# >>> Paddle preprocess
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



def _normalize_all_amounts_in_pipe(pipe_text: str) -> str:
    s = (pipe_text or "")
    if not s.strip():
        return s

    out_lines = []
    for ln in s.splitlines():
        parts = [p.strip() for p in ln.split("|")]
        if not parts:
            continue

        # Là "ô tiền"?
        def _is_amt_cell(x: str) -> bool:
            t = (x or "").strip().replace(" ", "")
            return bool(re.fullmatch(r"-?\(?\d{1,3}(?:[.,]\d{3})+(?:\)?)*$", t)) or \
                   bool(re.fullmatch(r"-?\d{4,}$", t))

        # Bỏ token thuyết minh dính phía trước số (vd "5.2 1.107...")
        def _strip_leading_note_token(x: str) -> str:
            return re.sub(r"^\s*\d+(?:\([a-z]\))?\s+(?=-?\(?\d)", "", x.strip(), flags=re.I)

        # các ô là số
        amt_idxs = [i for i, v in enumerate(parts) if _is_amt_cell(v)]

        # chỉ chuẩn hóa 2 ô số bên phải nhất
        for j in amt_idxs[-2:]:
            raw = _strip_leading_note_token(parts[j])
            raw_flat = raw.replace(",", ".").replace(" ", "")

            norm = normalize_vn_amount(raw)

            # Vá trường hợp thừa 1 chữ số đứng trước block số
            if not norm:
                if re.match(r"^[1-9]\d{4}\.", raw_flat):           # 6 + 7150...
                    norm = normalize_vn_amount(raw_flat[1:])
                elif re.match(r"^[1-9]\.\d{4}\.", raw_flat):        # 6.7150...
                    norm = normalize_vn_amount(raw_flat[2:])
                elif re.match(r"^[1-9]\d?\d?\.\d{3}\.", raw_flat):  # 61 150.941...
                    norm = normalize_vn_amount(raw_flat[1:])

            parts[j] = (norm or raw or "")


        out_lines.append(" | ".join(parts))

    return "\n".join(out_lines)


def _extract_inline_note_and_insert_col(pipe_text: str) -> str:
    """
    Nếu trong NAME có đuôi thuyết minh như '... 4', '... 5.2', '... 15.1(a)' thì
    tách nó ra thành 1 ô NOTE và CHÈN ngay trước cột tiền đầu tiên của chính dòng đó.
    Hàm này chỉ tác động trên từng dòng; không ép tăng K toàn bảng.
    """
    s = (pipe_text or "").strip()
    if not s: 
        return s

    lines = [ln for ln in s.splitlines() if ln.strip()]
    if not lines: 
        return s
    header, body = lines[0], lines[1:]

    def _is_amt(x):
        t = (x or "").replace(" ", "")
        return bool(re.fullmatch(r"-?\(?\d{1,3}(?:[.,]\d{3})+(?:\)?)*$", t)) or bool(re.fullmatch(r"-?\d{4,}$", t))
    def _is_code(x):
        return bool(re.fullmatch(r"\d{3}(?:\.\d+)?", (x or "").strip()))

    out = []
    for ln in body:
        parts = [p.strip() for p in ln.split("|")]
        if not parts:
            continue

        # cột tiền đầu tiên
        amt_idxs = [i for i,p in enumerate(parts) if _is_amt(p)]
        if not amt_idxs:
            out.append(" | ".join(parts))
            continue
        k0 = amt_idxs[0]

        # cần có CODE ở bên trái
        if not any(_is_code(p) for p in parts[:min(k0,3)]):
            out.append(" | ".join(parts))
            continue

        # ghép NAME (mọi thứ giữa CODE và cột tiền đầu)
        left = parts[:k0]
        code = left[0] if left else ""
        name = " ".join(left[1:]).strip()
        note = ""

        # bắt NOTE ở CUỐI tên
        m = re.search(r"(.*?)(\d+(?:\.\d+)?(?:\([a-z]\))?)\s*$", name, flags=re.I)
        if m:
            note = m.group(2).strip()
            name = m.group(1).strip()

        if not note:
            out.append(" | ".join(parts))
            continue

        # chèn NOTE ngay trước cột tiền đầu tiên (không thêm cột mới ở bên phải)
        new_cells = [code, name, note] + parts[k0:]
        # giữ placeholder trong cột HIỆN HỮU
        new_cells = [(c if str(c).strip() != "" else NULL_TOKEN) for c in new_cells]
        out.append(" | ".join(new_cells))

    # Header: chỉ chèn nhãn nếu chưa có
    h = [c.strip() for c in header.split("|")]
    if not any("uyết" in x.lower() or "note" in x.lower() for x in h):
        # chèn sau cột thứ hai (CODE|NAME|NOTE|…)
        if len(h) < 2:
            h += [""] * (2 - len(h))
        h.insert(2, "Thuyết minh")
        header = " | ".join(h)

    return "\n".join([header] + out)



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

# ====== Prefilter (header/caption mạnh) ======
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
    for i, ln in enumerate(lines):
        # bỏ hẳn các dòng chỉ có ký tự phân cách rác
        if re.fullmatch(r"[|\\\s]+", ln):
            continue

        parts = [p.strip() for p in ln.split("|")]
        if len(parts) < 2:
            continue
        low_all = _strip_vn_accents(ln)
        if i == 0:                # GIỮ header dòng đầu
            out.append(ln); continue
        if any(re.search(pat, low_all, re.I) for pat in _HEADER_STRONG_PATTERNS):
            continue
        if _DATE_LINE.search(low_all) or _VND_LINE.search(low_all):
            continue
        out.append(ln)

    return "\n".join(out)

def _trim_to_table_body(pipe_text: str) -> str:
    """
    Bỏ mọi dòng trước 'thân bảng': dòng đầu có mã 3 số và có ≥1 cột tiền ở bên phải.
    Dùng cho trang MIXED để không kéo caption/header vào bảng.
    """
    if not pipe_text.strip():
        return pipe_text
    lines = [ln for ln in pipe_text.splitlines() if ln.strip()]
    if not lines:
        return pipe_text

    def _is_amt(x): 
        t=(x or "").replace(" ","")
        return bool(re.fullmatch(r"-?\(?\d{1,3}(?:[.,]\d{3})+(?:\)?)*$", t)) or bool(re.fullmatch(r"-?\d{4,}$", t))
    def _is_code(x): 
        return bool(re.fullmatch(r"\d{3}(?:\.\d+)?", (x or "").strip()))

    start = 1  # bỏ header "C1|C2|..."
    for i in range(1, len(lines)):
        parts = [p.strip() for p in lines[i].split("|")]
        if not parts: 
            continue
        has_code = any(_is_code(p) for p in parts[:3])  # thường nằm bên trái
        amt_idxs = [j for j,p in enumerate(parts) if _is_amt(p)]
        if has_code and amt_idxs:
            start = i
            break

    keep = [lines[0]] + lines[start:]
    return "\n".join(keep)

def _looks_like_vas_5col(pipe_text: str) -> bool:
    """Nhận dạng header VAS có 'Mã số' / 'Thuyết minh' / 'Số cuối năm' / 'Số đầu năm'."""
    s = (pipe_text or "").lower()
    return ("ma so" in s or "mã số" in s) and \
           ("thuyet minh" in s or "thuyết minh" in s) and \
           ("so cuoi nam" in s or "số cuối năm" in s) and \
           ("so dau nam" in s or "số đầu năm" in s)

def _enforce_fin_5cols_if_header_missing(pipe_text: str) -> str:
    """
    Nếu bảng có header kiểu VAS nhưng pipe hiện tại chỉ 4 cột
    (bị gộp mất cột 'Thuyết minh'), tự chèn 1 cột trống C3 sau C2.
    """
    if not pipe_text.strip():
        return pipe_text

    lines = [ln for ln in pipe_text.splitlines() if ln.strip()]
    # đếm cột của dòng đầu tiên (header)
    cols0 = [c.strip() for c in lines[0].split("|")]
    n0 = len(cols0)

    if n0 >= 5:
        return pipe_text  # đủ rồi

    # chỉ áp dụng khi nhận diện header VAS
    if not _looks_like_vas_5col(pipe_text):
        return pipe_text

    fixed = []
    for ln in lines:
        parts = [p.strip() for p in ln.split("|")]
        if len(parts) == 4:
            # chèn 1 cột trống tại vị trí 2 (sau C2)
            parts = parts[:2] + [""] + parts[2:]
        elif len(parts) < 4:
            # các dòng ngắn thì pad cho đủ 5
            parts = parts + [""] * (5 - len(parts))
        fixed.append(" | ".join(parts))
    return "\n".join(fixed)



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

# ====== Parser/Score (giữ để đánh giá chất lượng nếu cần) ======
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

# ==============Narrator theo từng dòng
def _guess_semantic_cols(header: List[str]) -> Dict[str, int]:
    """
    Từ header pipe (C1..Cn hoặc text), đoán vị trí:
    code, name, note(thuyet minh), end, begin.
    Heuristic: lấy 2 cột số ở rìa phải làm end/begin; cột đầu là mã nếu trông như '100/131/151.1'.
    """
    n = len(header)
    idx_end = idx_begin = None
    # end/begin = 2 cột numeric ở phải
    idxs = list(range(n))
    # ưu tiên 2 cột cuối
    idx_cands = idxs[-3:]
    if len(idx_cands) >= 2:
        idx_end, idx_begin = idx_cands[-2], idx_cands[-1]
    # code/name/note
    idx_code = 0 if n >= 1 else None
    idx_name = 1 if n >= 2 else None
    idx_note = 2 if n >= 3 else None
    return {"code": idx_code, "name": idx_name, "note": idx_note, "end": idx_end, "begin": idx_begin}


# --- Helper: làm sạch narrator, bỏ '|' và rác dòng
def _post_clean_narrator(s: str) -> str:
    """
    - Bỏ ký tự '|' và rác cuối dòng, gộp khoảng trắng
    - Loại dòng trống/duplikate
    - Trả về chuỗi narrator dạng câu, mỗi dòng một câu
    """
    if not s:
        return ""

    import re as _re

    out = []
    for ln in (s.splitlines() or []):
        t = (ln or "").strip()
        if not t:
            continue
        # bỏ dấu '|' -> thay bằng ' – ' để đọc tự nhiên (muốn dùng dấu phẩy thì đổi thành ', ')
        t = _re.sub(r"\s*\|\s*", " – ", t)
        # bỏ dấu '\' hoặc ';' lẻ cuối dòng
        t = _re.sub(r"[\\;]\s*$", "", t)
        # gộp khoảng trắng thừa
        t = _re.sub(r"\s{2,}", " ", t)
        # chuẩn hoá dấu gạch đầu câu nếu có
        t = _re.sub(r"^\-+\s*", "– ", t)
        if t:
            out.append(t)

    # Khử trùng lặp theo thứ tự
    seen, dedup = set(), []
    for t in out:
        if t in seen:
            continue
        seen.add(t)
        dedup.append(t)

    return "\n".join(dedup).strip()



def build_narrator_from_pipe(pipe: str) -> str:
    """
    Narrator dựa trên suy luận cột tự động:
    - Không phụ thuộc header OCR
    - Hỗ trợ 0/1/2+ cột tiền
    """
    model = pipe_to_flex_schema(pipe or "")
    rows = model.get("rows") or []
    if not rows:
        return ""

    out = []
    for r in rows:
        head = " – ".join([x for x in [r.get("CODE",""), r.get("NAME","")] if x]).strip(" –")
        if not head: head = "(không tên)"
        parts = [head]
        if r.get("NOTE"): parts.append(f"TM: {r['NOTE']}")
        if r.get("END"):  parts.append(f"Cuối năm: {r['END']}")
        if r.get("BEGIN"):parts.append(f"Đầu năm: {r['BEGIN']}")
        out.append(" | ".join(parts))
    return _post_clean_narrator("\n".join(out))



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

# ====== Dò vạch kẻ dọc (rulings) để lock cột ======
def _detect_vertical_rulings_x(gray: np.ndarray) -> List[int]:
    """
    Trả về danh sách toạ độ x (pixel) của các vạch dọc chạy dài (>= 60% chiều cao trang).
    """
    if len(gray.shape) == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY_INV, 31, 15)
    h, w = gray.shape[:2]
    kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(25, h // 20)))
    vert = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kv, iterations=1)
    xs = []
    cnts, _ = cv2.findContours(vert, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        x, y, ww, hh = cv2.boundingRect(c)
        if hh >= int(0.6 * h) and ww <= max(2, w // 500):
            xs.append(x + ww // 2)
    if not xs:
        return []
    xs.sort()
    merged = []
    tol = max(6, w // 200)
    cur = [xs[0], xs[0]]
    for xi in xs[1:]:
        if xi - cur[1] <= tol:
            cur[1] = xi
        else:
            merged.append(int((cur[0] + cur[1]) // 2))
            cur = [xi, xi]
    merged.append(int((cur[0] + cur[1]) // 2))
    return merged

def _build_table_with_rulings(pil: Image.Image, ocr_lang: str, y_tol: int) -> Optional[str]:
    """
    Nếu tìm được >=4 vạch dọc → chia cột theo khoảng giữa các vạch (n-1 cột),
    group dòng theo y, rồi nhét từ theo cx vào từng cột.
    """
    bgr = cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    xs = _detect_vertical_rulings_x(gray)
    if len(xs) < 4:
        return None
    h, w = gray.shape[:2]
    bounds = [0] + sorted(xs) + [w]
    tsv = pytesseract.image_to_data(bgr, lang=ocr_lang, config=OCR_CFG_TSV, output_type=TessOutput.DICT)
    df = _tsv_dict_to_df(tsv)
    if df.empty:
        return ""
    rows = _cluster_rows(df, y_tol=y_tol)
    def _col_idx(cx):
        for i in range(len(bounds) - 1):
            if bounds[i] <= cx < bounds[i+1]:
                return i
        return len(bounds) - 2
    ncols = len(bounds) - 1
    header = [f"C{i+1}" for i in range(ncols)]
    lines = [" | ".join(header)]
    for row_df in rows:
        buckets = [[] for _ in range(ncols)]
        for _, t in row_df.iterrows():
            s = _norm_word(str(t["text"]))
            if not s: continue
            i = _col_idx(float(t["cx"]))
            buckets[i].append(s)
        cells = [" ".join(b).strip() for b in buckets]
        lines.append(" | ".join(cells))
    pipe = "\n".join(lines)
    return _prefilter_table_lines(pipe, {})

# ====== TABLE build pipelines (VAR-COLS) ======
def build_table_tsv(pil: Image.Image, y_tol: int, ocr_lang: str) -> Tuple[str, Dict[str,float]]:
    """
    Dựng bảng var-cols bằng TSV + KMeans + heuristics:
    - Ưu tiên k=5 nếu phát hiện header 5 cột (không ép; chỉ ưu tiên).
    - Ghép các cột text bị tách đôi trước khi xuất.
    - Chuẩn hóa: đảm bảo mọi hàng có cùng số cột = max_k quan sát.
    """
    from sklearn.metrics import silhouette_score

    bgr = cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    tsv = pytesseract.image_to_data(bgr, lang=ocr_lang, config=OCR_CFG_TSV, output_type=TessOutput.DICT)
    df = _tsv_dict_to_df(tsv)
    if df.empty:
        return "", {"score": 0.0}

    rows = _cluster_rows(df, y_tol=y_tol)
    var_rows: List[List[str]] = []
    want_k_global: Optional[int] = None

    # Thử phát hiện header 5 cột trên 3 hàng đầu
    probe_words = []
    for row_df in rows[:3]:
        toks = [str(t).strip() for t in row_df.sort_values("cx")["text"].tolist() if str(t).strip()]
        probe_words.append(toks)
    if any(_looks_like_5col_header(toks) for toks in probe_words):
        want_k_global = 5  # ƯU TIÊN (không ép)

    max_cols_seen = 0

    for row_df in rows:
        row_df = row_df[row_df["text"].astype(str).str.strip().astype(bool)]
        if row_df.empty:
            continue

        # —— KMeans 1D theo cx ——
        X = row_df[["cx"]].to_numpy(dtype=float)
        texts = [str(t).strip() for t in row_df.sort_values("cx")["text"].tolist()]

        if len(X) <= 2:
            cells = [" ".join(texts)]
            var_rows.append(cells)
            max_cols_seen = max(max_cols_seen, len(cells))
            continue

        k_best, labels_best, score_best = None, None, -1.0
        k_min, k_max = 2, min(9, len(X))

        # nếu có want_k_global thì thử trước
        k_trial = list(range(k_min, k_max+1))
        if want_k_global and want_k_global in k_trial:
            k_trial = [want_k_global] + [k for k in k_trial if k != want_k_global]

        for k in k_trial:
            try:
                km = KMeans(n_clusters=k, n_init="auto", random_state=0).fit(X)
                lab = km.labels_
                if len(set(lab)) < 2:  # k vô nghĩa
                    continue
                s = silhouette_score(X, lab)
                # heurstic boost: nếu k==5 và có >=2 cột số ở rìa phải → cộng điểm
                tmp_df = row_df.copy(); tmp_df["col_id"] = lab
                centers = tmp_df.groupby("col_id")["cx"].mean().sort_values()
                order = {cid:i for i,cid in enumerate(centers.index.tolist())}
                tmp_df["col_id"] = tmp_df["col_id"].map(order)
                buckets = {cid: [] for cid in sorted(tmp_df["col_id"].unique())}
                for _, t in tmp_df.sort_values(["col_id","cx"]).iterrows():
                    buckets[int(t["col_id"])].append(str(t["text"]).strip())
                cols = [" ".join(buckets[cid]).strip() for cid in sorted(buckets.keys())]
                right_nums = sum(1 for c in cols[-3:] if _is_num_cell(c))
                bonus = 0.15 if (k == 5 and right_nums >= 2) else 0.0
                s_eff = s + bonus
                if s_eff > score_best:
                    k_best, labels_best, score_best = k, lab, s_eff
            except Exception:
                continue

        if labels_best is None:
            cells = [" ".join(texts)]
            var_rows.append(cells)
            max_cols_seen = max(max_cols_seen, len(cells))
            continue

        row_df2 = row_df.copy(); row_df2["col_id"] = labels_best
        centers = row_df2.groupby("col_id")["cx"].mean().sort_values()
        order = {cid:i for i,cid in enumerate(centers.index.tolist())}
        row_df2["col_id"] = row_df2["col_id"].map(order)

        buckets = {cid: [] for cid in sorted(row_df2["col_id"].unique())}
        for _, t in row_df2.sort_values(["col_id","cx"]).iterrows():
            s = _norm_word(str(t["text"]))
            if s: buckets[int(t["col_id"])].append(s)

  
        cells = [" ".join(buckets[cid]).strip() for cid in sorted(buckets.keys())]

        # 1) ghép text bị tách đôi trước 2 cột số phải
        cells = _merge_adjacent_text_columns(cells, want_k_global)

        # 2) tách trường hợp 2 số tiền dính trong 1 cell → (END, BEGIN)
        cells = _split_two_amounts_in_cell(cells)

        var_rows.append(cells)




        max_cols_seen = max(max_cols_seen, len(cells))

    # Nếu đã ưu tiên header 5 cột và max<5 nhưng vẫn có 2 cột số bên phải ⇒ đẩy max lên 5
    if want_k_global == 5 and max_cols_seen < 5:
        # kiểm tra nhanh trên vài hàng
        for cells in var_rows[:5]:
            if sum(1 for c in cells[-3:] if _is_num_cell(c)) >= 2:
                max_cols_seen = max(max_cols_seen, 5)

    # Chuẩn hóa số cột trên toàn bảng
    K = max_cols_seen
    header = [f"C{i+1}" for i in range(K)]
    lines = [" | ".join(header)]
    for cells in var_rows:
        lines.append(" | ".join(_pad_to_k(cells, K)))

    pipe = "\n".join(lines)

    # chỉ vá khi nhìn giống header VAS và header đang có đúng 4 cột
    if _looks_like_vas_5col(pipe) and len((pipe.splitlines() or [''])[0].split("|")) == 4:
        pipe = _enforce_fin_5cols_if_header_missing(pipe) 

    pipe = _prefilter_table_lines(pipe, {})  # lọc caption mạnh
    return pipe, {"score": 1.0, "cols": K}


def build_table_paddle(pil: Image.Image, paddle_lang: str, paddle_gpu: bool) -> Tuple[str, Dict[str,float]]:
    pipe = paddle_table_to_pipe(pil, lang=paddle_lang, use_gpu=paddle_gpu) or ""
    rows_struct = parse_pipe_to_rows(pipe)
    metrics = score_table_quality(rows_struct)
    return pipe, metrics

# ---- Narrator ngắn ----
def _simple_narrator(pipe_text: str) -> str:
    if not pipe_text:
        return ""
    lines = [ln for ln in pipe_text.splitlines() if ln.strip()]
    if not lines:
        return ""
    nrows = max(0, len(lines) - 1)
    ncols = len(lines[0].split("|"))
    nums = 0
    roman = []
    for ln in lines[1:]:
        parts = [p.strip() for p in ln.split("|")]
        # giữ nguyên placeholder None (đừng strip thành rỗng)
        parts = [ (p if p.strip() != "" else NULL_TOKEN) for p in parts ]


        nums += sum(1 for p in parts if re.fullmatch(r"-?\d{1,3}(?:\.\d{3})+", p or ""))
        if re.match(r"^\s*(I|II|III|IV|V|VI|VII|VIII|IX|X)\.?\s", ln):
            roman.append(ln.split("|", 1)[0].strip())
    msg = [f"Bảng có {ncols} cột, {nrows} dòng dữ liệu."]
    if nums > nrows:
        msg.append("Nhận diện có các cột số tiền.")
    if roman:
        # unique giữ thứ tự
        runiq = list(dict.fromkeys(roman))
        msg.append("Mục chính: " + ", ".join(runiq)[:180])
    return " ".join(msg)

# ====== QUY TRÌNH 1 TRANG (VAR-COLS, KHÔNG ÉP 5 CỘT) ======
def process_page(
    pil: Image.Image, yaml_table: dict, yaml_text: dict, ocr_lang: str,
    use_gpt: bool, gpt_table_mode: str, gpt_model: str, gpt_temp: float,
    log_gpt: bool, gpt_scope: str = "table_only",
    do_autofix: bool = True, force_table: bool = False,
    rebuild_table: str = "auto", y_tol: int = 8,
    ocr_engine: str = "auto", table_engine: str = "auto",
    paddle_lang: str = "vi", paddle_gpu: bool = False,
    narrator_on: bool = True, table_format: str = "pipe",
    gpt_table_enabled: bool = True,
    prelight_mode: str = "auto",
    prelight_use: str = "auto",
    prelight_log: bool = False,
    roi_mask_top_ratio: float = 0.12,     # mask ~12% phía trên khi tìm ROI
    roi_mask_bot_ratio: float = 0.10,     # mask ~10% phía dưới
    roi_min_area_ratio: float = 0.03      # ngưỡng diện tích ROI
) -> Tuple[str, str, dict, List[Tuple[str,str]]]:
    """
    Trả về: (block_type, block_text, block_meta, extra_blocks)
    """
    extra_blocks: List[Tuple[str, str]] = []
    gpt_used = False

    # === PRELIGHT trước khi OCR/ROI ===
    pil_for_ocr, pl_meta, pil_bin = _prelight_apply(
        pil, mode=prelight_mode, choose_layer=prelight_use, log=prelight_log
    )

    # --- OCR TEXT cho trang (để nhận diện trang/table, lấy meta cơ bản) ---
    try:
        if ocr_engine == "paddle" or (ocr_engine == "auto" and _HAS_PADDLE and False):
            ocr_txt, meta_json = ocr_image_text_paddle(pil_for_ocr, lang=paddle_lang, use_gpu=paddle_gpu)
        else:
            ocr_txt, meta_json = ocr_image_text(pil_for_ocr, lang=ocr_lang)
    except Exception as e:
        print(f"[WARN] OCR engine failed ({ocr_engine}): {e} → fallback Tesseract")
        ocr_txt, meta_json = ocr_image_text(pil_for_ocr, lang=ocr_lang)

    meta = json.loads(meta_json) if meta_json else {}
    meta["_table_format"] = "var-col"
    meta["prelight"] = pl_meta

    # --- Phát hiện ROI bảng để tách TEXT/TABLE ---
    extra_blocks_mixed: List[Tuple[str, str]] = []
    try:
        _pil_for_roi = pil_bin if pil_bin is not None else pil_for_ocr
        _bgr0 = cv2.cvtColor(np.array(_pil_for_roi.convert("RGB")), cv2.COLOR_RGB2BGR)

        # mask top/bottom band để tránh header/footer “ăn” ROI
        H, W = _bgr0.shape[:2]
        if roi_mask_top_ratio > 0:
            cv2.rectangle(_bgr0, (0, 0), (W, int(H * roi_mask_top_ratio)), (255,255,255), -1)
        if roi_mask_bot_ratio > 0:
            cv2.rectangle(_bgr0, (0, int(H * (1.0 - roi_mask_bot_ratio))), (W, H), (255,255,255), -1)

        rois = _find_table_rois(_bgr0, min_area_ratio=roi_min_area_ratio)
    except Exception:
        rois = []

    # Log diện tích bảng (tham khảo)
    try:
        H2, W2 = _bgr0.shape[:2]
        page_area = float(H2 * W2)
        roi_area = sum((x2 - x1) * (y2 - y1) for (x1, y1, x2, y2) in rois) if rois else 0.0
        meta["roi_area_ratio"] = round(roi_area / max(1.0, page_area), 4)
    except Exception:
        meta["roi_area_ratio"] = None

    # Helper: chọn giữa TSV và Paddle theo độ “đầy” hoặc khi TSV trống
    def choose_better_pipe(tsv_pipe: str, pad_pipe: str) -> Tuple[str, str]:
        tsv_lines = tsv_pipe.strip().count("\n") if tsv_pipe else 0
        pad_lines = pad_pipe.strip().count("\n") if pad_pipe else 0
        if pad_lines > tsv_lines:
            return pad_pipe or "", "paddle"
        if tsv_lines > 0:
            return tsv_pipe or "", "tsv"
        return (pad_pipe or tsv_pipe or ""), ("paddle" if pad_pipe else "tsv")


    # ====== NHÁNH: MIXED PAGE (có ROI) ======
    if rois and not force_table:
        # 1) TEXT: che vùng bảng nhưng **giữ header**
        try:
            H = pil_for_ocr.height
            rois_text_mask = _guard_header_for_text_mask(rois, H, guard_ratio=0.12)
            pil_text_only = _mask_out_rois(pil_for_ocr, rois_text_mask)
            txt_only, _meta_text = ocr_image_text(pil_text_only, lang=ocr_lang)
            block_text_text = apply_yaml_clean(txt_only, yaml_text, mode="text").strip()

            # Fallback: nếu TEXT vẫn ngắn → OCR riêng dải header
            if not block_text_text or len(block_text_text) < 6:
                top = min(int(H * 0.22), min([r[1] for r in rois]) if rois else int(H * 0.22))
                top = max(120, top)
                pil_header = pil_for_ocr.crop((0, 0, pil_for_ocr.width, top))
                txt_head, _ = ocr_image_text(pil_header, lang=ocr_lang)
                block_text_text = (block_text_text + "\n" + txt_head).strip()

            # Giảm caption/bộ khung còn sót
            block_text_text = re.sub(r"(?im)^\s*(mã\s*số|ma\s*so|thuyết\s*minh|thuyet\s*minh|vnd|vnđ|đơn vị|don vi).*$", "", block_text_text)
            block_text_text = re.sub(r"\n{3,}", "\n\n", block_text_text).strip()
        except Exception:
            block_text_text = ""

        meta.setdefault("roi_tables", [])

        # 2) Với mỗi ROI: build TABLE var-col
        gl = (yaml_table.get("globals") or {}) if isinstance(yaml_table, dict) else {}
        pad_top = int(gl.get("roi_pad_top", 120))
        pad_lr  = int(gl.get("roi_pad_lr", 8))

        # Thu pipe theo từng ROI, lát nữa chọn cái “đầy dòng” nhất
        tsv_candidates: List[Tuple[str, int]] = []
        pad_candidates: List[Tuple[str, int]] = []
        rul_pipe = ""
        pil_roi_first = None

        for (x1, y1, x2, y2) in rois:
            y1_p = max(0, int(y1) - pad_top)
            x1_p = max(0, int(x1) - pad_lr)
            x2_p = min(pil_for_ocr.width,  int(x2) + pad_lr)
            y2_p = min(pil_for_ocr.height, int(y2))

            pil_roi = pil_for_ocr.crop((x1_p, y1_p, x2_p, y2_p))
            if pil_roi_first is None:
                pil_roi_first = pil_roi

            meta.setdefault("roi_padded", []).append([x1_p, y1_p, x2_p, y2_p])

            # a) TSV var-col (ứng viên)
            _tsv_pipe, _ = build_table_tsv(pil_roi, y_tol=y_tol, ocr_lang=ocr_lang)
            if _tsv_pipe and _tsv_pipe.strip():
                tsv_candidates.append((_tsv_pipe, _tsv_pipe.strip().count("\n")))

            # b) Paddle var-col (ứng viên)
            if _HAS_PADDLE and table_engine in ("auto", "paddle"):
                try:
                    _pad_pipe = paddle_table_to_pipe(pil_roi, lang=paddle_lang, use_gpu=paddle_gpu) or ""
                    if _pad_pipe and _pad_pipe.strip():
                        pad_candidates.append((_pad_pipe, _pad_pipe.strip().count("\n")))
                except Exception as e:
                    meta.setdefault("paddle_table_error", str(e))

        # Chọn pipe “đầy dòng” nhất cho mỗi phương án
        tsv_pipe = max(tsv_candidates, key=lambda x: x[1])[0] if tsv_candidates else ""
        pad_pipe = max(pad_candidates, key=lambda x: x[1])[0] if pad_candidates else ""

        # c) chọn route (ưu tiên theo đặc trưng trang) + rulings (vạch dọc)
        def _detect_grid_hint(bgr_page: np.ndarray) -> bool:
            gray = cv2.cvtColor(bgr_page, cv2.COLOR_BGR2GRAY)
            thr  = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            h, w = thr.shape[:2]
            kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, h // 40)))
            vert     = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel_v, iterations=1)
            frac = float((vert > 0).sum()) / max(1, h * w)
            return frac > 0.001

        # rulings: dùng ROI đầu tiên (nếu có) hoặc cả trang
        try:
            _pil_for_rulings = pil_roi_first if pil_roi_first is not None else pil_for_ocr
            rul_pipe = _build_table_with_rulings(_pil_for_rulings, ocr_lang=ocr_lang, y_tol=y_tol) or ""
        except Exception:
            rul_pipe = ""

        bgr_page = cv2.cvtColor(np.array(pil_for_ocr.convert("RGB")), cv2.COLOR_RGB2BGR)
        has_grid = _detect_grid_hint(bgr_page)

        # so sánh “độ đầy” theo số dòng
        def _nl(s): return s.strip().count("\n") if s else 0
        n_tsv  = _nl(tsv_pipe)
        n_pad  = _nl(pad_pipe)
        n_rul  = _nl(rul_pipe)

        # Cho phép Paddle “thua nhẹ” tối đa 2 dòng khi phát hiện grid
        grid_margin = 2

        if has_grid and n_pad > 0 and (n_pad + grid_margin) >= max(n_tsv, n_rul):
            chosen_pipe, route = pad_pipe, "paddle(grid)"
        elif n_rul > 0 and n_rul >= max(n_tsv, n_pad):
            chosen_pipe, route = rul_pipe, "rulings(lock-x)"
        elif n_tsv > 0 and n_tsv >= n_pad:
            chosen_pipe, route = tsv_pipe, "tsv(cluster)"
        else:
            chosen_pipe, route = (pad_pipe or tsv_pipe or rul_pipe or ""), (
                "paddle" if pad_pipe else ("tsv" if tsv_pipe else "rulings")
            )

        # === Build final pipe (auto, không ép cứng)
        block_text = (chosen_pipe or "").strip()
        block_text = _trim_to_table_body(block_text)
        block_text = _prefilter_table_lines(block_text, yaml_table)

        # 1) TÁCH NOTE nội dòng ra cột riêng (5.2, 7(a), 15.1...)
        block_text = _extract_inline_note_and_insert_col(block_text)
        block_text = _normalize_all_amounts_in_pipe(block_text)
        block_text = _repack_pipe_by_inferred_schema(block_text)   # KHÔNG truyền keyword



        # giữ phần auto schema + narrator như bạn đã có
        _auto = pipe_to_flex_schema(block_text)
        meta["auto_schema_indices"] = _auto.get("indices")

        # Narrator theo từng dòng (full-page)
        if narrator_on:
            nar = build_narrator_from_pipe(block_text)
            if nar and nar.strip():
                extra_blocks.append(("NARRATOR", nar))

        meta.update({
            "gpt_used": bool(gpt_used),
            "table_route": route,
            "text_sha1": _sha1(block_text),
            "engine": meta.get("engine"),
            "table_format": "var-col",
        })
        return "TABLE", block_text, meta, extra_blocks


    # ====== NHÁNH: TEXT (không có bảng) ======
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
                     table_format: str = "pipe",
                     gpt_table_enabled: bool = True,
                     prelight_mode: str = "auto",
                     prelight_use: str = "auto",
                     prelight_log: bool = False,
                     roi_mask_top_ratio: float = 0.12,
                     roi_mask_bot_ratio: float = 0.10,
                     roi_min_area_ratio: float = 0.03
                     ) -> None:

    print(f"📄 Input: {file_path}")
    out_txt, out_meta = make_output_paths(input_root, output_root, file_path)

    blocks: List[str] = []
    blocks_text_only: List[str] = []
    blocks_table_only: List[str] = []
    page_metas: List[dict] = []
    vector_chunks: List[dict] = []   # === JSONL chunks cho vector store

    total_pages = 0
    for page_idx, pil in iter_pages(file_path, dpi=dpi):
        if start_page is not None and page_idx + 1 < start_page:
            continue
        if end_page is not None and page_idx + 1 > end_page:
            continue
        total_pages += 1

        # [MOD] truyền gpt_table_enabled & prelight & ROI params theo CLI
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
            table_format=table_format,
            gpt_table_enabled=gpt_table_enabled,
            prelight_mode=prelight_mode,
            prelight_use=prelight_use,
            prelight_log=prelight_log,
            roi_mask_top_ratio=roi_mask_top_ratio,
            roi_mask_bot_ratio=roi_mask_bot_ratio,
            roi_min_area_ratio=roi_min_area_ratio
        )

        header = f"### [PAGE {page_idx+1:02d}] [{btype}]"
        blocks.append(f"{header}\n{block}\n")
        if btype == "TEXT":
            blocks_text_only.append(f"{header}\n{block}\n")
        else:
            blocks_table_only.append(f"{header}\n{block}\n")

        # === Build vector chunks + sample preview/log ===
        base_meta = {
            "page": page_idx + 1,
            "block_type": btype,
            "source_path": os.path.abspath(file_path),
            "engine": meta.get("engine"),
            "route": meta.get("table_route") or meta.get("engine") or None,
        }

        if btype == "TABLE":
            ch = _make_chunk(block, {**base_meta, "roi": None})
            if ch:
                vector_chunks.append(ch)
                _tbl_lines = [ln for ln in block.splitlines() if ln.strip()]
                if len(_tbl_lines) >= 2:
                    print("   [TABLE] sample:", _tbl_lines[1][:20])
            if "paddle_table_error" in meta:
                print("   [TABLE][WARN] paddle_table_error:", meta["paddle_table_error"])
        else:
            ch = _make_chunk(block, {**base_meta, "roi": None})
            if ch:
                vector_chunks.append(ch)

        for (bt2, bl2) in extra_blocks:
            header2 = f"### [PAGE {page_idx+1:02d}] [{bt2}]"
            blocks.append(f"{header2}\n{bl2}\n")
            blocks_table_only.append(f"{header2}\n{bl2}\n")
            if bt2.startswith("TABLE"):
                ch = _make_chunk(bl2, {**base_meta, "block_type": bt2, "roi": True})
                if ch:
                    vector_chunks.append(ch)
                    _tbl_lines2 = [ln for ln in bl2.splitlines() if ln.strip()]
                    if len(_tbl_lines2) >= 2:
                        print("   [TABLE-ROI] sample:", _tbl_lines2[1][:20])

        meta["page"] = page_idx + 1
        meta["block_type"] = btype
        page_metas.append(meta)

    final_text = "\n".join(blocks).strip()
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(final_text)
    print(f"📝 Wrote TXT: {out_txt}")

    try:
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

    # === Write vector-store JSONL ===
    out_vec = os.path.splitext(out_txt)[0] + "_vector.jsonl"
    with open(out_vec, "w", encoding="utf-8") as f:
        for ch in vector_chunks:
            if not ch or not ch.get("content"):
                continue
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")
    print(f"🧠 Wrote VEC: {out_vec}  (JSONL; mỗi dòng=1 chunk)")

    if split_debug:
        stem = os.path.splitext(out_txt)[0]
        if blocks_text_only:
            with open(stem + "_TEXT.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(blocks_text_only).strip())
        if blocks_table_only:
            with open(stem + "_TABLE.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(blocks_table_only).strip())
        print("🔎 Split-debug files written.")

def make_output_paths(input_root: str, output_root: str, file_path: str) -> Tuple[str,str]:
    rel = os.path.relpath(file_path, start=input_root)
    rel_dir = os.path.dirname(rel)
    stem = os.path.splitext(os.path.basename(file_path))[0]
    out_dir = os.path.join(output_root, rel_dir); ensure_dir(out_dir)
    return (os.path.join(out_dir, f"{stem}_text.txt"),
            os.path.join(out_dir, f"{stem}_meta.json"))

def build_argparser():
    p = argparse.ArgumentParser(
        "P1A (GPT) — Hybrid Auto-Route (OpenCV+TSV → Paddle) [VAR-COLS + PRELIGHT]"
    )
    # I/O
    p.add_argument("--input-root", type=str, default=INPUT_ROOT_DEFAULT)
    p.add_argument("--out-root",   type=str, default=OUTPUT_ROOT_DEFAULT)
    p.add_argument("--yaml-table", type=str, default=YAML_TABLE_DEFAULT)
    p.add_argument("--yaml-text",  type=str, default=YAML_TEXT_DEFAULT)

    # OCR / engines
    p.add_argument("--ocr-lang", type=str, default=OCR_LANG_DEFAULT)
    p.add_argument("--ocr-engine", choices=["auto","tesseract","paddle"], default="tesseract")
    p.add_argument("--table-engine", choices=["auto","tsv","paddle"], default="tsv")
    p.add_argument("--paddle-lang", type=str, default="vi")
    p.add_argument("--paddle-gpu", action="store_true")
    p.add_argument("--dpi", type=int, default=360, help="DPI render PDF/DOCX (khuyên 360–420)")
    p.add_argument("--y-tol", type=int, default=18, help="Y tolerance khi gom dòng TSV")

    # PRELIGHT
    p.add_argument("--prelight", choices=["auto","on","off"], default="auto",
                   help="Deskew + crop + adaptive bin. auto: bật khi cần; on: luôn bật; off: tắt")
    p.add_argument("--prelight-use", choices=["auto","bin","gray"], default="auto",
                   help="Chọn lớp đưa vào OCR/ROI: auto/bin/gray")
    p.add_argument("--prelight-log", action="store_true", help="In log quyết định prelight cho từng trang")

    # ROI tuning (NEW)
    p.add_argument("--roi-mask-top", type=float, default=0.12,
                   help="Tỉ lệ chiều cao trang cần mask phía trên khi dò ROI (0..0.5).")
    p.add_argument("--roi-mask-bottom", type=float, default=0.10,
                   help="Tỉ lệ chiều cao trang cần mask phía dưới khi dò ROI (0..0.5).")
    p.add_argument("--roi-min-area", type=float, default=0.03,
                   help="Ngưỡng diện tích tối thiểu của ROI so với trang (mặc định 0.03 = 3%).")

    # Output / formatting
    p.add_argument("--table-format", choices=["ascii","pipe","json"], default="pipe",
                   help="Định dạng xuất bảng: ascii, pipe, json (var-col vẫn pipe)")
    p.add_argument("--split-debug", action="store_true")
    p.add_argument("--start", type=int, default=None)
    p.add_argument("--end",   type=int, default=None)
    p.add_argument("--narrator", choices=["y","n"], default="y")

    # Rebuild / validator toggles (giữ nguyên để tương thích)
    p.add_argument("--rebuild-table", choices=["auto","none","force"], default="auto")
    p.add_argument("--no-autofix", action="store_true")
    p.add_argument("--force-table", action="store_true")

    # GPT toggles
    p.add_argument("--no-gpt", action="store_true",
                   help="Tắt toàn bộ bước dùng GPT")
    p.add_argument("--gpt-table", choices=["y","n"], default="y",
                   help="Bật/Tắt GPT cho xử lý bảng var-col (y/n). Mặc định: y")
    p.add_argument("--gpt-table-mode", choices=["financial","generic","auto"], default="financial")
    p.add_argument("--gpt-model", type=str, default="gpt-4o-mini")
    p.add_argument("--gpt-temp", type=float, default=0.0)
    p.add_argument("--log-gpt", action="store_true",
                   help="In log đầu ra GPT rút gọn để debug")
    p.add_argument("--gpt-scope", choices=["table_only","all","none"], default="table_only",
                   help="Phạm vi GPT: chỉ bảng / cả text & bảng / không dùng")

    return p

# ====== CLI ======
def main():
    args = build_argparser().parse_args()

    # === Chuẩn hoá page range (1-based) ===
    if args.start is not None and args.start <= 0:
        print("⚠️ --start phải ≥ 1 → auto set = 1")
        args.start = 1
    if args.end is not None and args.end <= 0:
        print("⚠️ --end phải ≥ 1 → auto set = 1")
        args.end = 1

    # Nếu chỉ có --start mà không có --end → hiểu là 1 trang
    if args.start is not None and args.end is None:
        args.end = args.start

    # Nếu nhập ngược → hoán đổi
    if args.start is not None and args.end is not None and args.start > args.end:
        print(f"ℹ️ Đổi phạm vi trang {args.start}..{args.end} → {args.end}..{args.start}")
        args.start, args.end = args.end, args.start

    # --- Áp dụng MASTER SWITCH ---
    if GPT_MASTER_SWITCH is True:
        eff_no_gpt = False          # ép bật GPT
        eff_gpt_table_flag = "y"
        master_note = "FORCED-ON"
    elif GPT_MASTER_SWITCH is False:
        eff_no_gpt = True           # ép tắt GPT
        eff_gpt_table_flag = "n"
        master_note = "FORCED-OFF"
    else:
        eff_no_gpt = args.no_gpt    # theo CLI như cũ
        eff_gpt_table_flag = getattr(args, "gpt_table", "y")
        master_note = "CLI"

    gpt_table_enabled = (eff_gpt_table_flag == "y" and not eff_no_gpt)
    gpt_enhancer_state = "READY" if globals().get("_HAS_GPT_ENHANCER", False) else "MISSING"

    # --- Summary cấu hình ---
    print("========== P1A RUN CONFIG ==========")
    print(f"[INPUT ] input-root : {os.path.abspath(args.input_root)}")
    print(f"[OUTPUT] out-root   : {os.path.abspath(args.out_root)}")
    print(f"[ENGINE] OCR={args.ocr_engine} | TABLE={args.table_engine} | DPI={args.dpi}")
    print(f"[PREL  ] prelight={args.prelight} | use={args.prelight_use} | log={'ON' if args.prelight_log else 'OFF'}")
    print(f"[ROI   ] mask_top={args.roi_mask_top} | mask_bottom={args.roi_mask_bottom} | min_area={args.roi_min_area}")
    print(f"[FORMAT] table-format={args.table_format} | narrator={'ON' if args.narrator=='y' else 'OFF'}")
    print(f"[GPT   ] enabled={('NO' if eff_no_gpt else 'YES')} | mode={args.gpt_table_mode} | model={args.gpt_model} | source={master_note}")
    print(f"[GPT.T ] TABLE={'ON' if gpt_table_enabled else 'OFF'} (scope={args.gpt_scope}, enhancer={gpt_enhancer_state})")
    print(f"[YAML ] Trạng thái: {'ĐANG TẮT (RAW MODE)' if P1A_RAW_MODE else 'ĐANG BẬT (dùng YAML)'}")
    if not P1A_RAW_MODE:
        print(f"[YAML ] yaml-table : {args.yaml_table}")
        print(f"[YAML ] yaml-text  : {args.yaml_text}")
    print("====================================")

    if args.start is None and args.end is None:
        print("[PAGES] all pages")
    else:
        print(f"[PAGES] range: {args.start}..{args.end} (1-based, inclusive)")

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

    # ==== Hiệu lực cờ GPT (đặt trước vòng for) ====
    gpt_table_flag = getattr(args, "gpt_table", "y")
    if 'GPT_MASTER_SWITCH' in globals() and GPT_MASTER_SWITCH is True:
        eff_no_gpt = False
        eff_gpt_table_flag = "y"
    elif 'GPT_MASTER_SWITCH' in globals() and GPT_MASTER_SWITCH is False:
        eff_no_gpt = True
        eff_gpt_table_flag = "n"
    else:
        eff_no_gpt = args.no_gpt
        eff_gpt_table_flag = gpt_table_flag
    gpt_table_enabled = (eff_gpt_table_flag == "y" and not eff_no_gpt)

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
                use_gpt=(not eff_no_gpt),
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
                narrator_on=(args.narrator == "y"),
                dpi=args.dpi,
                table_format=args.table_format,
                gpt_table_enabled=gpt_table_enabled,
                prelight_mode=args.prelight,
                prelight_use=args.prelight_use,
                prelight_log=args.prelight_log,
                roi_mask_top_ratio=max(0.0, min(0.5, args.roi_mask_top)),
                roi_mask_bot_ratio=max(0.0, min(0.5, args.roi_mask_bottom)),
                roi_min_area_ratio=max(0.005, min(0.5, args.roi_min_area))  # chặn biên cho an toàn
            )

        except Exception as e:
            import traceback
            print(f"❌ Lỗi file: {fp} → {e}")
            traceback.print_exc()

    print("\n✅ Hoàn tất. Mỗi file input sinh ra: TXT + META + VECTOR.jsonl.")
    print("   Bật --split-debug để có thêm file _TEXT/_TABLE; dùng --narrator n để tắt diễn giải khi QA.")
    print(f"📌 Toàn bộ output đang nằm dưới thư mục: {os.path.abspath(args.out_root)}")

if __name__ == "__main__":
    main()
