# -*- coding: utf-8 -*-
"""
src/p1a_clean10_ocr_bctc.py — OCR từ ảnh prelight (ưu tiên _bin.png) → TXT + META

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
import os, re, glob, json, argparse, hashlib
from typing import Optional, Tuple, Dict, List

import numpy as np
import cv2
from PIL import Image
import pytesseract
from pytesseract import Output as TessOutput
import yaml

# [ADD] hỗ trợ clean/append
import shutil
APPEND_MODE = False  # sẽ bật True trong main() khi --clean a

import src.env  # ✅ đảm bảo nạp .env.active và set OPENAI_API_KEY
from src.gpt_enhancer import enhance_with_gpt

# ========= ĐƯỜNG DẪN MẶC ĐỊNH (theo yêu cầu) =========
PRELIGHT_DIR_DEFAULT = r"D:\1.TLAT\3. ChatBot_project\1_Insurance_Strategy\outputs\p1_prelight_ocr_bctc"
ORIG_DIR_DEFAULT     = r"D:\1.TLAT\3. ChatBot_project\1_Insurance_Strategy\inputs\c_financial_reports_test"
OUTPUT_DIR_DEFAULT   = r"D:\1.TLAT\3. ChatBot_project\1_Insurance_Strategy\outputs\p1a_clean10_ocr_bctc_GPT-90"
YAML_PATH = r"D:\1.TLAT\3. ChatBot_project\1_Insurance_Strategy\configs\configs\rules_table_universal.yaml"

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

def _looks_like_table(text: str) -> bool:
    if not text:
        return False
    patt = r"(mã\s*số|ma\s*so|chỉ\s*tiêu|chi\s*tieu|số\s*cuối\s*năm|so\s*cuoi\s*nam|số\s*đầu\s*năm|so\s*dau\s*nam)"
    return re.search(patt, text, re.IGNORECASE) is not None


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
    if not text:
        return None, None

    norm_all = _norm_letters(text)

    # Balance sheet (chịu lỗi OCR ở "balance    sheet")
    if re.search(r"b[a]?ng\s+can\s+do[i1l]\s+ke\s+toan", norm_all):
        return "Bảng cân đối kế toán", "balance_sheet"
    if (re.search(r"balance\s*sheet", norm_all)
        or "statement of financial position" in norm_all):
        return "Bảng cân đối kế toán", "balance_sheet"

    # Income statement (thêm profit & loss / statement of profit or loss)
    if ("bao cao ket qua hoat dong kinh doanh" in norm_all
        or "income statement" in norm_all
        or re.search(r"profit\s*(and|&)\s*loss", norm_all)
        or "statement of profit or loss" in norm_all):
        return "Báo cáo kết quả hoạt động kinh doanh", "income_statement"

    # Cash flow (chịu “cashflows/cash flow”)
    if ("bao cao luu chuyen tien te" in norm_all
        or re.search(r"cash\s*flow(s)?", norm_all)):
        return "Báo cáo lưu chuyển tiền tệ", "cash_flow"

    # Changes in equity
    if ("bao cao thay doi von chu so huu" in norm_all
        or "changes in equity" in norm_all):
        return "Báo cáo thay đổi vốn chủ sở hữu", "equity_changes"

    # Heuristic: nếu trong text có >=2 tín hiệu bảng BCTC → coi là balance sheet
    keys = 0
    for k in ["ma so", "so cuoi nam", "so dau nam", "chi tieu"]:
        if k in norm_all:
            keys += 1
    if keys >= 2:
        return "Bảng cân đối kế toán", "balance_sheet"

    # Fallback: lấy dòng dài nhất trong 15 dòng đầu làm tiêu đề
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


# hàm tìm file gốc theo base
def find_original_for_base(base: str, orig_root: str) -> Optional[str]:
    """Tìm file gốc (PDF/DOCX/XLSX/...) có stem == base trong orig_root (đệ quy)."""
    if not orig_root or not os.path.isdir(orig_root):
        return None
    exts = ["pdf", "docx", "doc", "xlsx", "xls", "pptx", "ppt"]
    for ext in exts:
        for path in glob.glob(os.path.join(orig_root, "**", f"*.{ext}"), recursive=True):
            try:
                if os.path.splitext(os.path.basename(path))[0] == base:
                    return path
            except Exception:
                continue
    return None

def process_one_page(out_root: str, base: str, page_no: int,
                     src_img_path: str, ocr_lang: str, ocr_cfg: str,
                     source_pdf: Optional[str] = None) -> None:
    out_dir = os.path.join(out_root)  # p1a đặt phẳng theo yêu cầu
    ensure_dir(out_dir)

    # debug đầu hàm
    print(f"→ process_one_page: base={base}, page={page_no:02d}, img={os.path.basename(src_img_path)}", flush=True)

    # OCR
    bgr = cv2.cvtColor(np.array(Image.open(src_img_path).convert("RGB")), cv2.COLOR_RGB2BGR)
    txt, meta_partial_json = ocr_image_to_text_and_meta(bgr, ocr_lang, ocr_cfg)
    meta_partial = json.loads(meta_partial_json)
    print(f"   OCR len={len(txt)}; looks_like_table={_looks_like_table(txt)}", flush=True)

    # === YAML cleanup/autofix ===
    def apply_yaml_autofix(text: str, yaml_path: str) -> str:
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                ycfg = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"⚠️ Không thể đọc YAML: {e}")
            return text

        gl  = (ycfg.get("globals") or {})
        ncl = (gl.get("number_cleanup") or {})

        # drop_chars
        for ch in ncl.get("drop_chars", []):
            text = text.replace(ch, "")
        # fix_patterns
        for pat in ncl.get("fix_patterns", []):
            fr, to = pat.get("from", ""), pat.get("to", "")
            try:
                text = re.sub(fr, to, text)
            except re.error:
                pass
        # thousand_grouping
        if ncl.get("thousand_grouping", False):
            text = re.sub(r"(?<=\d)[,](?=\d{3}\b)", ".", text)
            text = re.sub(r"(\d)\.(\d{1,2})\.(\d{3})(?=[^\d]|$)", r"\1\2.\3", text)

        # code_aliases (mức text phẳng)
        aliases = gl.get("code_aliases", {}) or {}
        for raw, ali in aliases.items():
            text = re.sub(rf"(?<!\d){re.escape(raw)}(?![\d])", ali, text)

        # autofix nhẹ nếu có
        for rule in (ycfg.get("autofix") or []):
            cond   = (rule.get("when") or "")
            action = (rule.get("do")   or "")
            m1 = re.search(r"startswith\(end,\s*'([0-9.]+)'\)", cond)
            m2 = re.search(r"replace_prefix\(end,\s*'(\d)','(\d)'\)", action)
            if m1 and m2 and m1.group(1) in text:
                old_d, new_d = m2.groups()
                text = text.replace(f"{old_d}.", f"{new_d}.", 1)
            alias_match = re.search(r"alias\('([^']+)'\)", action)
            if "raw_code" in cond and alias_match:
                old_code = alias_match.group(1)
                text = text.replace(old_code, aliases.get(old_code, old_code))
        return text

    # Đường dẫn YAML ĐÚNG (configs\...)
    YAML_PATH = r"D:\1.TLAT\3. ChatBot_project\1_Insurance_Strategy\configs\rules_table_universal.yaml"
    txt = apply_yaml_autofix(txt, YAML_PATH)

    # === GPT: chỉ khi là bảng & có API key ===
    if _looks_like_table(txt):
        if os.getenv("OPENAI_API_KEY"):
            print("   GPT: ON (table detected)", flush=True)
            txt = enhance_with_gpt(txt, meta_partial, src_img_path)
        else:
            print("   GPT: OFF (missing OPENAI_API_KEY)", flush=True)
    else:
        print("   GPT: SKIP (no table detected)", flush=True)

    # === GHI FILE (đặt BÊN TRONG HÀM) ===
    text_path = os.path.join(out_dir, f"{base}_page{page_no:02d}_text.txt")
    print(f"   → write text: {text_path}", flush=True)
    if APPEND_MODE and os.path.exists(text_path):
        print(f"↩️ Bỏ qua (append): {os.path.basename(text_path)} đã tồn tại")
    else:
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(txt)

    meta = {
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
        "preprocess": True,
        "ocr_lang": ocr_lang,
        "ocr_cfg": ocr_cfg,
        "text_sha1": _sha1_text(txt),
    }

    meta_path = os.path.join(out_dir, f"{base}_page{page_no:02d}_meta.json")
    print(f"   → write meta : {meta_path}", flush=True)
    if APPEND_MODE and os.path.exists(meta_path):
        print(f"↩️ Bỏ qua (append): {os.path.basename(meta_path)} đã tồn tại")
    else:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"📝 Saved: {os.path.basename(text_path)}, {os.path.basename(meta_path)}", flush=True)

    # dòng sau
    # (phần còn lại trong hàm hoặc kết thúc hàm)


def run_ocr_on_prelight(prelight_dir: str, out_dir: str,
                        start_page: int, end_page: Optional[int],
                        ocr_lang: str, ocr_cfg: str,
                        prefer: str = "bin",
                        orig_dir: Optional[str] = None) -> None:
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


    test_path = os.path.join(out_dir, "_write_test.tmp")
    try:
        with open(test_path, "w", encoding="utf-8") as _t:
            _t.write("ok")
        os.remove(test_path)
        print("✔️ Output folder writable")
    except Exception as _e:
        print(f"⛔ Không ghi được vào output: {out_dir} → {_e}")

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
        source_orig = find_original_for_base(base, orig_dir) if orig_dir else None
        try:
            process_one_page(out_root=out_dir, base=base, page_no=pg,
                            src_img_path=img_path, ocr_lang=ocr_lang, ocr_cfg=ocr_cfg,
                            source_pdf=source_orig)
            # verify files
            tp = os.path.join(out_dir, f"{base}_page{pg:02d}_text.txt")
            mp = os.path.join(out_dir, f"{base}_page{pg:02d}_meta.json")
            if os.path.exists(tp) and os.path.exists(mp):
                print(f"✅ DONE page {pg:02d} (created files)")
            else:
                print(f"❗ DONE page {pg:02d} nhưng chưa thấy file: "
                    f"{os.path.basename(tp) if not os.path.exists(tp) else ''} "
                    f"{os.path.basename(mp) if not os.path.exists(mp) else ''}")
        except Exception as e:
            import traceback
            print(f"❌ Lỗi khi xử lý {base}_page{pg:02d}: {e}")
            traceback.print_exc()


# ========= CLI =========
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("P1A — OCR từ ảnh prelight (_bin/_orig) → TXT + META")
    p.add_argument("--prelight-dir", type=str, default=PRELIGHT_DIR_DEFAULT,
                   help="Thư mục chứa ảnh prelight (mirror, có *_pageNN_bin.png / *_pageNN_orig.png)")
    p.add_argument("--orig-dir", type=str, default=ORIG_DIR_DEFAULT,
               help="Thư mục chứa file gốc (PDF/DOCX/XLSX/...) để ghi nguồn vào meta")
    p.add_argument("--out", type=str, default=OUTPUT_DIR_DEFAULT,
                   help="Thư mục output (đặt phẳng theo yêu cầu)")
    p.add_argument("--start", type=int, default=1, help="Trang bắt đầu")
    p.add_argument("--end", type=int, default=None, help="Trang kết thúc (inclusive)")
    p.add_argument("--prefer", choices=["bin","orig","auto"], default="bin",
                   help="Ưu tiên dùng ảnh nào (mặc định: bin)")
    p.add_argument("--ocr-lang", type=str, default=OCR_LANG_DEFAULT, help="Ngôn ngữ OCR (mặc định: vie+eng)")
    p.add_argument("--ocr-cfg",  type=str, default=OCR_CFG_DEFAULT,  help="Tesseract config (mặc định: --psm 6)")

    # [ADD] hỏi/xoá/append/bỏ qua khi output đã tồn tại
    p.add_argument("--clean", choices=["ask","y","a","n"], default="ask",
                help="ask: hỏi; y: xoá output cũ; a: giữ thư mục & chỉ ghi file mới; n: bỏ qua nếu đã tồn tại")
    return p


def main():
    global APPEND_MODE  # phải đứng TRƯỚC mọi phép gán APPEND_MODE trong hàm

    args = build_argparser().parse_args()

    # [ADD] Chuẩn bị thư mục output theo --clean
    out_dir = args.out
    if os.path.exists(out_dir):
        choice = args.clean
        if choice == "ask":
            choice = input(f"⚠️ Output '{out_dir}' đã tồn tại. y=xoá, a=append, n=bỏ qua: ").strip().lower()
        if choice == "y":
            shutil.rmtree(out_dir, ignore_errors=True); print(f"🗑️ Đã xoá {out_dir}")
        elif choice == "n":
            print("⏭️ Bỏ qua P1A."); return
        elif choice == "a":
            print(f"➕ Giữ {out_dir}, chỉ ghi file mới.")
        else:
            print("❌ Lựa chọn không hợp lệ → bỏ qua."); return

    os.makedirs(out_dir, exist_ok=True)

    # [READ YAML ONCE]
    YAML_PATH = r"D:\1.TLAT\3. ChatBot_project\1_Insurance_Strategy\configs\rules_table_universal.yaml"
    try:
        with open(YAML_PATH, "r", encoding="utf-8") as _f:
            YAML_RULES = yaml.safe_load(_f) or {}
    except Exception as _e:
        print(f"⚠️ Không đọc được YAML rule: {_e}")
        YAML_RULES = {}

    # [ADD] bật cờ append-only
    APPEND_MODE = (args.clean == "a")

    # Chạy OCR trên ảnh prelight

    run_ocr_on_prelight(
        prelight_dir=args.prelight_dir,
        out_dir=args.out,
        start_page=args.start,
        end_page=args.end,
        ocr_lang=args.ocr_lang,
        ocr_cfg=args.ocr_cfg,
        prefer=args.prefer,
        orig_dir=args.orig_dir
    )

    print("\n✅ Hoàn tất P1A. Kiểm tra *_text.txt và *_meta.json tại thư mục output.")

if __name__ == "__main__":
    main()


