# -*- coding: utf-8 -*-
"""
p2_balance_checks.py — Re-OCR 2 cột số + kiểm tra & tự sửa hàng tổng (BCTC VN)
🟢 Output mỗi trang CHỈ còn 2 file (giống P1):
  <base>_page{n}_text.txt      (bảng 5 cột, số đã được cân/sửa)
  <base>_page{n}_meta.json     (meta gốc + log p2_balance_checks)

Input  (từ P1):  .../_meta.json (bắt buộc), .../_text.txt (không bắt buộc)
Hỗ trợ:
- Mã số 2–3 chữ số và nhánh: 10, 111, 111.1, 131.1.2…
- Rule động theo mã số: X00 = X10..X90 ; XY0 = XY1..XY9 ; parent.* là con
- Rule tĩnh (nạp thêm bằng --rules) có thể chứa dấu: ["+10","-11","+111.1"]
- Tolerance: --abs-tol & --rel-tol; --dry-run chỉ báo cáo không sửa
Yêu cầu: pdf2image pillow pytesseract pandas numpy scikit-learn opencv-python-headless
"""

import os, re, json, glob, argparse
from typing import Optional, List, Dict, Tuple
import numpy as np
import pandas as pd

import cv2
from pdf2image import convert_from_path
import pytesseract
from sklearn.cluster import KMeans  # để dành nếu cần tinh ROI

# ======== cấu hình ========
NUM_OCR_CFG = "--oem 1 --psm 6 -c tessedit_char_whitelist=0123456789.,-()"
POPPLER_PATH = os.environ.get("POPPLER_PATH", None)
TESSERACT_CMD = os.environ.get("TESSERACT_CMD", None)
if TESSERACT_CMD and os.path.isfile(TESSERACT_CMD):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

IN_COLS = ["Mã số","Khoản mục/Tài sản","Thuyết minh","Số cuối năm","Số đầu năm"]
CODE_RE = re.compile(r"^\d{2,3}(?:\.\d+)*$")  # HỖ TRỢ 2–3 chữ số + nhánh
NOTE_TOKEN_RE = re.compile(r"^(?:[IVXLC]+|\d+(?:\.\d+){0,2})$", re.IGNORECASE)

DEFAULT_IN_ROOT  = os.path.join("outputs", "p1_clean10_orc_raw_output")   # theo yêu cầu: orc
DEFAULT_OUT_ROOT = os.path.join("outputs", "p2_clean10_orc_raw_output")

# ======== utils ========
def ensure_dir(p: str): os.makedirs(p, exist_ok=True)
def _is_code(s: str) -> bool: return bool(CODE_RE.fullmatch(str(s).strip()))
def _looks_note_token(s: str) -> bool: return bool(NOTE_TOKEN_RE.fullmatch(str(s).strip()))

def _normalize_num(s: Optional[str]) -> Optional[int]:
    if s is None:
        return None
    t = str(s)

    # 1) Sửa ký tự OCR thường nhầm
    trans = str.maketrans({
        "I": "1", "l": "1", "|": "1", "¡": "1",
        "O": "0", "o": "0",
        "S": "5", "£": "5",
        "B": "8",
    })
    t = t.translate(trans)

    # 2) Bỏ nháy cong/đơn
    t = (t.replace("“", "").replace("”", "")
           .replace("‘", "").replace("’", "")
           .replace("'", ""))

    # 3) Giữ lại chữ số, . , - và ngoặc ()
    t = re.sub(r"[^\d\.,\-\(\)]", "", t)

    # 4) Dấu âm kiểu (123) => âm
    neg = "(" in t and ")" in t
    t = t.replace("(", "").replace(")", "")

    # 5) Xử lý . và , lẫn lộn
    if t.count(".") and t.count(","):
        if t.count(".") >= t.count(","):
            t = t.replace(",", "")
        else:
            t = t.replace(".", "")

    digits = re.sub(r"[.,]", "", t)
    if re.fullmatch(r"-?\d+", digits):
        v = int(digits)
        return -v if neg else v
    return None

# ======== PDF helpers ========
def _render_page(meta: dict) -> np.ndarray:
    first = int(meta.get("page", 1))
    last  = int(meta.get("page", 1))
    dpi   = int(meta.get("dpi", 500))
    src   = meta["source_pdf"]
    kw = dict(dpi=dpi, first_page=first, last_page=last)
    if POPPLER_PATH: kw["poppler_path"] = POPPLER_PATH
    pil = convert_from_path(src, **kw)[0]
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

def _detect_table_roi(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    thr  = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    h, w = thr.shape
    vert = cv2.erode(thr, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, h//60))), 1)
    vert = cv2.dilate(vert, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, h//60))), 2)
    hori = cv2.erode(thr, cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, w//60), 1)), 1)
    hori = cv2.dilate(hori, cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, w//60), 1)), 2)
    grid = cv2.bitwise_or(vert, hori)
    contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return img_bgr
    cnt = max(contours, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(cnt)
    pad = 6
    y0=max(0,y-pad); x0=max(0,x-pad)
    y1=min(img_bgr.shape[0], y+h+pad); x1=min(img_bgr.shape[1], x+w+pad)
    return img_bgr[y0:y1, x0:x1]

# ======== OCR “words” & nhóm dòng ========
def _normalize_vn(s: str) -> str:
    t = str(s).lower()
    rep = {"â":"a","ă":"a","á":"a","à":"a","ả":"a","ã":"a","ạ":"a",
           "ê":"e","é":"e","è":"e","ẻ":"e","ẽ":"e","ẹ":"e",
           "ô":"o","ơ":"o","ó":"o","ò":"o","ỏ":"o","õ":"o","ọ":"o",
           "ư":"u","ú":"u","ù":"u","ủ":"u","ũ":"u","ụ":"u",
           "í":"i","ì":"i","ỉ":"i","ĩ":"i","ị":"i",
           "ý":"y","ỳ":"y","ỷ":"y","ỹ":"y","ỵ":"y","đ":"d"}
    for k,v in rep.items(): t=t.replace(k,v)
    return re.sub(r"\s+"," "," ".join(t.split()))

def _find_header_anchors(df_words: pd.DataFrame) -> Optional[Dict[str,Tuple[int,int]]]:
    if df_words.empty: return None
    w = df_words.copy()
    w["norm"] = w["text"].map(_normalize_vn)

    cand_note  = w[w["norm"].str.contains(r"\bthuyet\s*minh\b")]
    cand_end   = w[w["norm"].str.contains(r"\bso\s*cuoi(\s*nam)?\b")]
    cand_start = w[w["norm"].str.contains(r"\bso\s*dau(\s*nam)?\b")]
    if cand_note.empty or cand_end.empty or cand_start.empty: return None

    x_note  = float(cand_note.sort_values("left").iloc[0]["left"])
    x_end   = float(cand_end.sort_values("left").iloc[0]["left"])
    x_start = float(cand_start.sort_values("left").iloc[0]["left"])
    left_min = float(w["left"].min())

    code_right = left_min + (x_note - left_min) * 0.22
    item_right = x_note   - (x_note - code_right) * 0.35
    note_right = x_note   + (x_end  - x_note)     * 0.30
    end_halfw  = max(45, (x_start - x_end)        * 0.22)

    return {
        "code":  (0, int(code_right)),
        "item":  (int(code_right)+1, int(item_right)),
        "note":  (int(item_right)+1, int(note_right)),
        "end":   (max(0, int(x_end-end_halfw)),   int(x_end+end_halfw)),
        "start": (max(0, int(x_start-end_halfw)), int(x_start+end_halfw)),
    }

def _group_lines(df_words: pd.DataFrame) -> List[pd.DataFrame]:
    lines, cur, last_y = [], [], None
    if df_words.empty: return lines
    thr = max(8.0, float(df_words["height"].median()) * 0.9)
    for _, r in df_words.sort_values(["y_center","left"]).iterrows():
        y = r["y_center"]
        if last_y is None or abs(y - last_y) <= thr:
            cur.append(r)
        else:
            lines.append(pd.DataFrame(cur)); cur=[r]
        last_y = y
    if cur: lines.append(pd.DataFrame(cur))
    return lines

def _pre_num_img(crop_bgr: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.bilateralFilter(g, 7, 60, 60)
    g = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

def _ocr_numbers_only(crop_bgr: np.ndarray) -> str:
    img = _pre_num_img(crop_bgr)
    txt = pytesseract.image_to_string(img, lang="eng", config=NUM_OCR_CFG)
    return re.sub(r"[^\d\.,\-()]", "", txt or "").strip()

def _ocr_numbers_table(meta: dict) -> pd.DataFrame:
    from pytesseract import Output
    page = _render_page(meta)
    roi  = _detect_table_roi(page)
    data = pytesseract.image_to_data(roi, lang="eng+vie", output_type=Output.DATAFRAME).dropna(subset=["text"])
    data = data[data["text"].astype(str).str.strip().astype(bool)]
    if data.empty: return pd.DataFrame(columns=IN_COLS)
    data["y_center"] = data["top"] + data["height"]/2

    anchors = _find_header_anchors(data)
    lines   = _group_lines(data)

    out=[]
    for g in lines:
        row = ["","","","",""]
        y0 = int(g["top"].min()); y1 = int((g["top"]+g["height"]).max())
        for _, tok in g.sort_values("left").iterrows():
            t = str(tok["text"]).strip()
            lx = float(tok["left"])

            if anchors and lx <= anchors["code"][1] and _is_code(t):
                row[0] = (row[0]+" "+t).strip(); continue
            if anchors and anchors["item"][0] <= lx <= anchors["note"][1] and _looks_note_token(t):
                row[2] = (row[2]+" "+t).strip(); continue
            if anchors and lx <= anchors["note"][0]:
                row[1] = (row[1]+" "+t).strip(); continue

        # Re-OCR riêng 2 cột số
        if anchors:
            x0,x1 = anchors["end"];   crop = roi[max(0,y0-2):min(roi.shape[0],y1+2), x0:x1]
            row[3] = _ocr_numbers_only(crop)
            x0,x1 = anchors["start"]; crop = roi[max(0,y0-2):min(roi.shape[0],y1+2), x0:x1]
            row[4] = _ocr_numbers_only(crop)

        out.append(row)

    df = pd.DataFrame(out, columns=IN_COLS)
    # bỏ phần đầu cho tới dòng đầu có mã số
    first=None
    for i,v in enumerate(df["Mã số"]):
        if _is_code(v): first=i; break
    if first is not None: df = df.iloc[first:].reset_index(drop=True)

    for col in ["Số cuối năm","Số đầu năm"]:
        df[col] = df[col].map(_normalize_num)

    blank = df["Mã số"].astype(str).str.strip().eq("") \
            & df["Khoản mục/Tài sản"].astype(str).str.strip().eq("") \
            & df["Số cuối năm"].isna() & df["Số đầu năm"].isna()
    return df[~blank].reset_index(drop=True)

# ======== RULES ========
def _present_codes(df: pd.DataFrame) -> set:
    return set(df["Mã số"].dropna().astype(str).str.strip())

def _has_prefix(present: set, code: str) -> bool:
    return (code in present) or any(c.startswith(code + ".") for c in present)

def dynamic_children_for(parent: str, present: set) -> List[str]:
    """
    - X00 → {X10..X90} (có mặt mới lấy) + parent.*
    - XY0 → {XY1..XY9} (có mặt mới lấy) + parent.*
    - Else → parent.* (các nhánh con)
    """
    kids=set()
    if not re.fullmatch(r"\d{3}", parent):
        kids |= {c for c in present if c.startswith(parent + ".")}
        return sorted(kids)

    p = int(parent)
    if p % 100 == 0:
        base = (p // 100) * 100
        for t in range(base+10, base+100, 10):
            code = f"{t:03d}"
            if _has_prefix(present, code): kids.add(code)
    if p % 10 == 0:
        base = (p // 10) * 10
        for t in range(base+1, base+10):
            code = f"{t:03d}"
            if _has_prefix(present, code): kids.add(code)

    kids |= {c for c in present if c.startswith(parent + ".")}
    return sorted(kids)

def load_static_rules(path: Optional[str]) -> Dict[str, List[str]]:
    # BCĐKT mẫu (Tài sản + vài nhóm); có thể nạp ngoài để bổ sung NV, KQKD, LCTT…
    base = {
        "100": ["110","120","130","140","150","160","170","180","190"],
        "130": ["131","136","139"],
        "140": ["141"],
        "150": ["151","152"],
        "200": ["210","220","230","240","250","260","270","280","290"],
        "300": ["100","200"]
    }
    if path and os.path.isfile(path):
        try:
            with open(path,"r",encoding="utf-8") as f:
                user = json.load(f) or {}
            for k,v in user.items():
                base[k] = list(v)
        except Exception as e:
            print(f"⚠️ Không đọc được rules {path}: {e}")
    return base

def build_rules(
    df: pd.DataFrame,
    static_rules: Dict[str, List[str]]
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, List[str]]]:

    present = _present_codes(df)
    rules_used={}
    missing={}
    newkids={}
    for parent, stat_children in static_rules.items():
        dyn = dynamic_children_for(parent, present)
        merged = []
        seen=set()
        for c in dyn: merged.append(c); seen.add(c)
        for c in stat_children:
            code = c[1:] if c[:1] in ['+','-'] else c  # bỏ dấu khi merge danh sách hiển thị
            if code not in seen: merged.append(code)
        rules_used[parent] = merged
        stat_plain = [c[1:] if c[:1] in ['+','-'] else c for c in stat_children]
        missing[parent] = [c for c in stat_plain if not _has_prefix(present, c)]
        newkids[parent] = [c for c in dyn if c not in stat_plain]
    for parent in sorted(present):
        if parent not in rules_used:
            dyn = dynamic_children_for(parent, present)
            if dyn:
                rules_used[parent]=dyn
                missing[parent]=[]
                newkids[parent]=dyn
    return rules_used, missing, newkids

# ======== Signed children helpers ========
def _split_sign(child: str) -> Tuple[int, str]:
    s = str(child).strip()
    sign = -1 if s.startswith('-') else 1
    code = s[1:] if s[:1] in ['+','-'] else s
    return sign, code

def _match_children_index_signed(df: pd.DataFrame, children: List[str]) -> List[Tuple[int, pd.Index]]:
    col = df["Mã số"].fillna("").astype(str)
    out = []
    for c in children:
        sign, code = _split_sign(c)
        mask = col.str.fullmatch(re.escape(code)) | col.str.startswith(code + ".")
        idx = df.index[mask]
        if len(idx) > 0:
            out.append((sign, idx))
    return out

def _within_tolerance(observed: Optional[int], expected: int, abs_tol: int, rel_tol: float) -> bool:
    if observed is None: return False
    diff = abs(int(observed) - int(expected))
    if diff <= abs_tol: return True
    denom = max(1, abs(expected))
    return (diff / denom) <= rel_tol

def enforce_totals(df: pd.DataFrame,
                   rules_used: Dict[str,List[str]],
                   static_rules: Dict[str,List[str]],
                   abs_tol: int,
                   rel_tol: float,
                   dry_run: bool) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Khi parent có trong static_rules → dùng exactly danh sách static (giữ dấu nếu có),
    ngược lại dùng rules_used (toàn dấu '+', tức cộng).
    """
    out = df.copy()
    changes=[]
    for parent, children_auto in rules_used.items():
        where_parent = out["Mã số"].astype(str) == str(parent)
        if not where_parent.any(): 
            continue
        pid = out.index[where_parent][0]

        # con có dấu: ưu tiên static_rules; fallback '+' cho tất cả con động
        if parent in static_rules:
            children_signed = static_rules[parent]
        else:
            children_signed = ["+" + c for c in children_auto]

        parts = _match_children_index_signed(out, children_signed)

        for col in ["Số cuối năm","Số đầu năm"]:
            total = 0
            for sign, idx in parts:
                total += sign * pd.to_numeric(out.loc[idx, col], errors="coerce").fillna(0).sum()
            obs = out.at[pid, col]
            if not _within_tolerance(obs, int(total), abs_tol, rel_tol):
                changes.append({
                    "code": parent, "column": col,
                    "observed": None if pd.isna(obs) else int(obs),
                    "suggested": int(total),
                    "children_used": children_signed
                })
                if not dry_run:
                    out.at[pid, col] = int(total)
    return out, changes

# ======== ONLY-P2 OUTPUT (text + meta) ========
def _fmt_int(v):
    return "" if pd.isna(v) else f"{int(v):,}".replace(",", ".")

def dataframe_to_text(df: pd.DataFrame) -> str:
    """
    Xuất lại text 5 cột (tab-separated) theo schema:
    Mã số | Khoản mục/Tài sản | Thuyết minh | Số cuối năm | Số đầu năm
    """
    lines = []
    header = ["Mã số","Khoản mục/Tài sản","Thuyết minh","Số cuối năm","Số đầu năm"]
    lines.append("\t".join(header))
    for _, r in df.iterrows():
        row = [
            str(r.get("Mã số","") or "").strip(),
            str(r.get("Khoản mục/Tài sản","") or "").strip(),
            str(r.get("Thuyết minh","") or "").strip(),
            _fmt_int(r.get("Số cuối năm")),
            _fmt_int(r.get("Số đầu năm")),
        ]
        lines.append("\t".join(row))
    return "\n".join(lines)

def write_text_and_meta(df_fixed: pd.DataFrame, meta: dict, changes: list, out_dir: str, base: str):
    # 1) *_text.txt
    text_path = os.path.join(out_dir, f"{base}_text.txt")
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(dataframe_to_text(df_fixed))

    # 2) *_meta.json (giữ meta + thêm log P2)
    meta2 = dict(meta)
    meta2.setdefault("postprocess", {})
    meta2["postprocess"]["p2_balance_checks"] = {
        "status": "ok" if len(changes)==0 else "warning",
        "mismatches": len(changes),
        "changes": changes
    }
    meta_path = os.path.join(out_dir, f"{base}_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta2, f, ensure_ascii=False, indent=2)
    return text_path, meta_path

# ======== One-page pipeline ========
def _apply_unit_multiplier(df: pd.DataFrame, mul: int) -> pd.DataFrame:
    out = df.copy()
    for col in ["Số cuối năm","Số đầu năm"]:
        out[col] = out[col].map(lambda x: None if x is None else int(x) * int(mul))
    return out

def process_one(meta_path: str,
                out_root: str,
                rules_path: Optional[str],
                abs_tol: int,
                rel_tol: float,
                dry_run: bool):
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # mirror thư mục từ in_root sang out_root
    src_dir = os.path.dirname(meta_path)
    try:
        rel = os.path.relpath(src_dir, start=os.path.commonpath([src_dir]))
    except Exception:
        rel = os.path.basename(src_dir)
    out_dir = os.path.join(out_root, rel)
    ensure_dir(out_dir)

    # 1) Re-OCR bảng số theo cột
    df_raw = _ocr_numbers_table(meta)

    # 2) Áp đơn vị
    mul = int(meta.get("unit_multiplier") or 1)
    df_raw = _apply_unit_multiplier(df_raw, mul)

    # 3) Xây rule (tĩnh + động)
    static_rules = load_static_rules(rules_path)
    rules_used, missing, newkids = build_rules(df_raw, static_rules)

    # 4) Kiểm tra & (tuỳ) sửa totals
    df_fixed, changes = enforce_totals(df_raw, rules_used, static_rules, abs_tol, rel_tol, dry_run)

    # 5) Ghi đúng 2 file (text + meta)
    base = os.path.basename(meta_path).replace("_meta.json", "")
    text_path, meta_path_out = write_text_and_meta(df_fixed, meta, changes, out_dir, base)
    print(f"✅ {base}: wrote\n  - {text_path}\n  - {meta_path_out}")

# ======== CLI ========
def main():
    ap = argparse.ArgumentParser("Balance checks & auto-fix totals (static + dynamic, signed rules) — TEXT+META only")
    ap.add_argument("--in",   dest="in_dir",  default=DEFAULT_IN_ROOT,
                    help=f"Thư mục chứa *_meta.json từ p1 (mặc định: {DEFAULT_IN_ROOT})")
    ap.add_argument("--out",  dest="out_dir", default=DEFAULT_OUT_ROOT,
                    help=f"Thư mục xuất mirror kết quả (mặc định: {DEFAULT_OUT_ROOT})")
    ap.add_argument("--rules",dest="rules",   default=None,
                    help="JSON rule tĩnh bổ sung (parent -> [children]; child có thể '+120'/'-22')")
    ap.add_argument("--abs-tol", dest="abs_tol", type=int, default=0,
                    help="Ngưỡng sai số tuyệt đối cho phép (mặc định 0)")
    ap.add_argument("--rel-tol", dest="rel_tol", type=float, default=0.0,
                    help="Ngưỡng sai số tương đối cho phép (vd 1e-6 ~ 0.0001%)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Chỉ báo cáo chênh lệch, KHÔNG sửa dữ liệu")
    args = ap.parse_args()

    metas = glob.glob(os.path.join(args.in_dir, "**", "*_meta.json"), recursive=True)
    if not metas:
        print("❌ Không tìm thấy *_meta.json trong", args.in_dir); return

    for mp in metas:
        try:
            process_one(mp, args.out_dir, args.rules, args.abs_tol, args.rel_tol, args.dry_run)
        except Exception as e:
            print(f"⚠️ Lỗi xử lý {mp}: {e}")

if __name__ == "__main__":
    main()
