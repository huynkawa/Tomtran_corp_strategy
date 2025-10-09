# p1_clean10_version3-92% (v3-92%: phân tách phần bảng & phần văn bản)
"""
Mục tiêu: phần không phải bảng xuất RAW (văn bản thường), phần bảng xuất theo 5 cột căn lề:
[Mã số | Chỉ tiêu | Thuyết minh | Số cuối năm | Số đầu năm]

Chiến lược:
- Dùng OpenCV tìm vùng bảng dựa trên line detection (morphology horizontal/vertical) → bbox lớn nhất.
- OCR hai phần riêng: ngoài bảng = `image_to_string` (raw); trong bảng = pipeline 5 cột với tìm ranh giới cột (tiêu đề/K-means), gom dòng, làm sạch số.
- Hợp nhất đầu ra: RAW (ngoài bảng) + ASCII table (trong bảng). Nếu không phát hiện bảng → fallback v6 (toàn bộ là bảng) + header RAW rỗng.
"""

import os, re, cv2, json, unicodedata, hashlib
import numpy as np
import pytesseract
from pytesseract import Output
from pdf2image import convert_from_path
from typing import List, Tuple, Optional, Dict

# ===== Cấu hình =====
TESSERACT_LANG = "vie+eng"
TESSERACT_CFG  = "--psm 6 -c preserve_interword_spaces=1"
DPI = 500
EXT_IMAGE = [".jpg", ".jpeg", ".png"]
INPUT_DIR_DEFAULT  = r"inputs/a1_p1_prelight_ocr_bctc_SCAN"
OUTPUT_DIR_DEFAULT = r"outputs/p1_clean10_version3_92"

# ===== Utils =====
def ensure_dir(p): os.makedirs(p, exist_ok=True)
def _sha1(s: str) -> str: return hashlib.sha1(s.encode("utf-8")).hexdigest()

def preprocess_image(bgr):
    gray  = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (3,3), 0)
    th    = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,21,8)
    return th  # ảnh GRAY nhị phân để tìm bảng

def to_bgr(gray_or_bgr):
    if len(gray_or_bgr.shape)==2: return cv2.cvtColor(gray_or_bgr, cv2.COLOR_GRAY2BGR)
    return gray_or_bgr

def strip_vn_accents(s: str) -> str:
    s = (s or "").replace("Đ","D").replace("đ","d")
    s = unicodedata.normalize("NFD", s)
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn")

def norm_vn(s: str) -> str:
    s = strip_vn_accents(s).lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+"," ",s).strip()

NUM_TOKEN = re.compile(r"[0-9][0-9\.,]{2,}")

def clean_number(s: str) -> str:
    if not s: return ""
    s = (s or "").replace("O","0").replace("o","0").replace("Ô","0").replace("U","0")
    s = s.replace(",", "").replace(" ", "")
    s = re.sub(r"[^0-9\.]", "", s)
    s = re.sub(r"\.{2,}", ".", s)
    return s.strip(".")

# ===== Tìm vùng bảng bằng morphology =====
def find_table_bbox(gray: np.ndarray) -> Optional[Tuple[int,int,int,int]]:
    H,W = gray.shape[:2]
    inv = 255 - gray
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, W//40), 1))
    vert_kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, H//40)))
    horiz = cv2.morphologyEx(inv, cv2.MORPH_OPEN, horiz_kernel, iterations=2)
    vert  = cv2.morphologyEx(inv, cv2.MORPH_OPEN, vert_kernel,  iterations=2)
    table_mask = cv2.add(horiz, vert)
    table_mask = cv2.dilate(table_mask, cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)), iterations=2)
    cnts,_ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    areas = [(cv2.contourArea(c), c) for c in cnts]
    areas.sort(key=lambda x: x[0], reverse=True)
    for area, c in areas:
        x,y,w,h = cv2.boundingRect(c)
        if w*h > 0.1*W*H and w > 0.4*W and h > 0.25*H:
            return x,y,w,h
    return None

# ===== Ranh giới 2 cột số =====
def find_columns_by_headers(words: List[Dict], page_h: int) -> Optional[Tuple[int,int]]:
    top_cut = int(page_h * 0.4)
    head = [w for w in words if w['top'] < top_cut]
    if not head: return None
    by_line: Dict[int, List[Dict]] = {}
    for w in head: by_line.setdefault(w.get('line_num',0), []).append(w)
    end_x = start_x = None
    for toks in by_line.values():
        toks.sort(key=lambda x: x['left'])
        text = " ".join(t['text'] for t in toks)
        n = norm_vn(text)
        if "so cuoi nam" in n: end_x  = min(t['left'] for t in toks)
        if "so dau nam"  in n: start_x = min(t['left'] for t in toks)
    if end_x is not None and start_x is not None:
        split1 = max(min(end_x, start_x) - 30, 180)   # tránh hẹp cột mã số
        split2 = (end_x + start_x) // 2
        return split1, max(split2, split1+120)
    return None

def _kmeans1d(xs: np.ndarray, iters=15):
    c1, c2 = np.percentile(xs,35), np.percentile(xs,70)
    for _ in range(iters):
        d1, d2 = np.abs(xs-c1), np.abs(xs-c2)
        g1, g2 = xs[d1<=d2], xs[d2<d1]
        if len(g1): c1 = float(np.mean(g1))
        if len(g2): c2 = float(np.mean(g2))
    split = (c1+c2)/2.0
    return float(min(c1,c2)), float(max(c1,c2)), float(split)

def find_columns_fallback_by_numbers(words: List[Dict], W: int) -> Tuple[int,int]:
    xs=[]
    for w in words:
        if NUM_TOKEN.search(w['text']):
            x = w['left'] + w['width']/2
            if x > W*0.45: xs.append(x)
    if len(xs)>=6:
        xs = np.array(xs, dtype=float)
        c1,c2,split = _kmeans1d(xs); split1 = int(min(c1,c2)-50); split2=int(split)
    else:
        split1 = int(W*0.58); split2 = int(W*0.76)
    split1 = max(180, split1); split2 = max(split1+120, split2)
    return split1, split2

# ===== Parse cột trái thành 3 thành phần =====
# Cho phép ký tự rác (| . - : +) trước mã số
_re_code = re.compile(r"^\s*[\+\|\.\-: ]{0,8}(\d{3})\b")
_re_note = re.compile(r"(?:^|\s)(\d{1,2}(?:\.\d)?)\s*$")   # 4, 5.2, 15.1,...

def parse_left_side(text: str) -> Tuple[str, str, str]:
    s = (text or "").strip()
    s = re.sub(r"^[\+\|\.\-: ]+", "", s)  # bỏ rác đầu chuỗi
    ma = chi = tm = ""
    m = _re_code.match(s)
    if m:
        ma = m.group(1); s = s[m.end():].strip(" .-:|+")
    m = _re_note.search(s)
    if m:
        tm = m.group(1); s = s[:m.start()].strip(" .-:|+")
    chi = s
    return ma, chi, tm

# ===== Helpers số & thuyết minh =====
_NUM_GROUP = re.compile(r"\d{1,3}(?:\.\d{3})+")

def is_note_token(text: str) -> bool:
    t = (text or "").strip()
    return bool(re.fullmatch(r"\d{1,2}(?:\.\d)?", t))

def _score_number(txt: str) -> tuple:
    digits = re.sub(r"\D", "", txt or "")
    return (txt.count(".") if txt else 0, len(digits))

def tokens_to_number(tokens: List[Dict]) -> str:
    s = " ".join(t["text"] for t in tokens)
    cands = _NUM_GROUP.findall(s)
    if cands:
        best = max(cands, key=_score_number)
        return clean_number(best)
    return clean_number(s)

def split_two_numbers_by_x(tokens: List[Dict]) -> Tuple[str, str]:
    nums = []
    for t in tokens:
        if _NUM_GROUP.search(t["text"]) or NUM_TOKEN.search(t["text"]):
            xmid = t["left"] + t["width"]/2
            nums.append((xmid, t["text"]))
    if len(nums) >= 2:
        nums.sort(key=lambda x: x[0])  # trái → phải
        left_txt  = max(_NUM_GROUP.findall(nums[0][1]) or [nums[0][1]], key=_score_number)
        right_txt = max(_NUM_GROUP.findall(nums[1][1]) or [nums[1][1]], key=_score_number)
        return clean_number(left_txt), clean_number(right_txt)
    return "", ""


def _pick_two_numbers_by_x(tokens: List[Dict], split1: int, split2: int) -> Tuple[str, str]:
    """
    Lấy tất cả ứng viên số ở bên phải split1, chấm điểm (nhiều dấu '.' và dài hơn),
    rồi chọn 2 ứng viên 'tốt' nhất nhưng vẫn tôn trọng vị trí trái/phải.
    Trả về (end_num, start_num) = (trái, phải).
    """
    cands = []
    for t in tokens:
        xmid = t['left'] + t['width'] / 2
        if xmid > (split1 - 6):
            for m in _NUM_GROUP.findall(t['text']):
                score = _score_number(m)
                cands.append((xmid, m, score))
    if not cands:
        return "", ""
    # Ưu tiên ứng viên có điểm cao; khi bằng nhau thì theo khoảng cách tới rìa trái/phải
    cands.sort(key=lambda x: (x[2], -abs(x[0]-split1)), reverse=True)
    # Lấy tối đa 3 ứng viên tốt nhất để còn dư địa chọn trái/phải
    top = cands[:3]
    # Sắp lại theo X trái→phải để gán (end,start)
    top.sort(key=lambda x: x[0])
    if len(top) == 1:
        # 1 số: nếu nằm bên phải split2 → gán cho start, else cho end
        x, m, _ = top[0]
        if x > split2: return "", clean_number(m)
        return clean_number(m), ""
    # >=2 số: lấy trái nhất và phải nhất
    left = clean_number(top[0][1])
    right = clean_number(top[-1][1])
    # Nếu hai số trùng nhau và còn ứng viên giữa → thử thay bằng ứng viên có điểm cao hơn
    if left == right and len(top) >= 3:
        # Chọn cặp có khoảng cách X lớn nhất nhưng giá trị khác nhau
        for i in range(len(top)-1):
            for j in range(len(top)-1, i, -1):
                a = clean_number(top[i][1]); b = clean_number(top[j][1])
                if a != b:
                    return a, b
    return left, right

# ===== Reflow trong bảng =====
def reflow_records(img_bgr: np.ndarray) -> List[Dict]:
    data = pytesseract.image_to_data(
        img_bgr, lang=TESSERACT_LANG, config=TESSERACT_CFG, output_type=Output.DICT
    )
    n = len(data['text'])
    words=[]
    for i in range(n):
        t = (data['text'][i] or "").strip()
        if not t: continue
        try: conf = float(data['conf'][i])
        except: conf = 0
        if conf < 0: continue
        words.append({
            "text": t,
            "left": int(data['left'][i]),
            "top": int(data['top'][i]),
            "width": int(data['width'][i]),
            "line_num": int(data.get('line_num',[0]*n)[i]),
            "par_num":  int(data.get('par_num',[0]*n)[i]),
            "block_num":int(data.get('block_num',[0]*n)[i]),
        })
    if not words: return []

    H, W = img_bgr.shape[:2]
    split1, split2 = (find_columns_by_headers(words, H) or
                      find_columns_fallback_by_numbers(words, W))

    # Gom dòng theo (block, par, line) và vá theo Y
    groups: Dict[Tuple[int,int,int], List[Dict]] = {}
    for w in words:
        key=(w['block_num'], w['par_num'], w['line_num'])
        groups.setdefault(key, []).append(w)

    raw=[]
    for toks in groups.values():
        toks.sort(key=lambda x:(x['top'], x['left']))
        raw.append(toks)
    raw.sort(key=lambda arr: arr[0]['top'])

    merged=[]
    for arr in raw:
        if not merged: merged.append(arr); continue
        last = merged[-1]
        if abs(arr[0]['top'] - last[-1]['top']) <= 6:
            last.extend(arr); last.sort(key=lambda t: t['left'])
        else:
            merged.append(arr)

    # tham số phân cột
    MARGIN  = 18      # đệm ranh giới
    TM_BAND = 120     # băng bắt số thuyết minh (mở rộng)

    rows=[]
    for toks in merged:
        toks.sort(key=lambda t:t['left'])
        left_tokens=[]; end_tokens=[]; start_tokens=[]
        tm_candidates=[]

        for t in toks:
            xmid = t['left'] + t['width']/2
            if   xmid < (split1 - MARGIN):
                left_tokens.append(t)
            elif xmid < (split2 - MARGIN):
                end_tokens.append(t)
            elif xmid > (split2 + MARGIN):
                start_tokens.append(t)
            else:
                # “vùng chết” → gán cột gần nhất
                if abs(xmid - split1) < abs(xmid - split2): end_tokens.append(t)
                else:                                       start_tokens.append(t)

            # Bắt số Thuyết minh ở dải quanh split1 (cả hai phía)
            if (split1 - TM_BAND) <= xmid <= (split1 + MARGIN) and is_note_token(t['text']):
                tm_candidates.append(t['text'])

        left_text = " ".join(t['text'] for t in left_tokens).strip()

        # bỏ số thuyết minh khỏi 2 cột tiền
        end_tokens_f   = [t for t in end_tokens   if not is_note_token(t['text'])]
        start_tokens_f = [t for t in start_tokens if not is_note_token(t['text'])]

        # số theo từng cột → chọn ứng viên tốt nhất
        end_num   = tokens_to_number(end_tokens_f)
        start_num = tokens_to_number(start_tokens_f)

        # một bên chứa cả 2 số → tách theo X
        if (not start_num and len(end_tokens_f) >= 2) or (not end_num and len(start_tokens_f) >= 2):
            e2, s2 = split_two_numbers_by_x(end_tokens_f + start_tokens_f)
            end_num   = end_num   or e2
            start_num = start_num or s2

        # Nếu 2 cột trùng nhau HOẶC thiếu 1 cột → chọn lại theo toàn bộ số bên phải split1
        if (end_num and start_num and end_num == start_num) or (end_num and not start_num) or (start_num and not end_num):
            e2, s2 = _pick_two_numbers_by_x(toks, split1, split2)
            # Gán khi có cải thiện (không trùng hoặc bổ sung được cột trống)
            if e2 or s2:
                if not end_num or (e2 and e2 != start_num):   end_num = e2 or end_num
                if not start_num or (s2 and s2 != end_num):   start_num = s2 or start_num

        # Số thuyết minh
        tm_text = ""
        if tm_candidates:
            tm_text = tm_candidates[0]
        else:
            for col_tokens in (end_tokens, start_tokens):
                small_notes = [t for t in col_tokens if is_note_token(t['text'])]
                if small_notes:
                    tm_text = small_notes[0]['text']; break

        ma, chi, tm = parse_left_side(left_text)
        if not tm and tm_text: tm = tm_text

        if ma or chi or tm or end_num or start_num:
            rows.append({"ma": ma, "chi": chi, "tm": tm, "end": end_num, "start": start_num})

    # gộp dòng tiếp diễn (không mã/tm/số) vào dòng trước (ví dụ: “đáo hạn”)
    fused = []
    for r in rows:
        if (fused and not r["ma"] and not r["tm"] and not r["end"] and not r["start"]
                and r["chi"] and len(r["chi"]) <= 28):
            fused[-1]["chi"] = (fused[-1]["chi"] + " " + r["chi"]).strip()
        else:
            fused.append(r)
    return fused

# ===== Xuất bảng căn lề =====
def format_table(rows: List[Dict]) -> str:
    headers = ["Mã số", "Chỉ tiêu", "Thuyết minh", "Số cuối năm", "Số đầu năm"]

    def fix_text(s: str) -> str:
        s = (s or "").strip()
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"^([a-zà-ỹ])", lambda m: m.group(1).upper(), s, flags=re.UNICODE)
        repl = {
            "tai bao hiem": "tái bảo hiểm",
            "hoa hong": "hoa hồng",
            "phai thu": "phải thu",
            "bao hiem": "bảo hiểm",
            "chi phi": "chi phí",
            "tien": "tiền",
            "thue": "thuế",
            "du phong": "dự phòng",
            "hang ton kho": "hàng tồn kho",
            "ngan han": "ngắn hạn",
            "dau tu": "đầu tư",
            "nam giu": "nắm giữ",
            "dao han": "đáo hạn",
            "iv.": "IV.", "vi.": "VI.", "v.": "V.", "iii.": "III.", "ii.": "II.", "i.": "I.",
        }
        low = strip_vn_accents(s).lower()
        for k,v in repl.items():
            if k in low:
                s = re.sub(re.escape(k), v, strip_vn_accents(s).lower()).replace(k, v)
        return s

    for r in rows:
        r["ma"]    = (r.get("ma","") or "").strip()
        r["chi"]   = fix_text(r.get("chi","") or "")
        r["tm"]    = (r.get("tm","") or "").strip()
        r["end"]   = (r.get("end","") or "").strip()
        r["start"] = (r.get("start","") or "").strip()

    w_ma    = max([len(r["ma"])    for r in rows] + [len(headers[0])]) if rows else len(headers[0])
    w_chi   = max([len(r["chi"])   for r in rows] + [len(headers[1])]) if rows else len(headers[1])
    w_tm    = max([len(r["tm"])    for r in rows] + [len(headers[2])]) if rows else len(headers[2])
    w_end   = max([len(r["end"])   for r in rows] + [len(headers[3])]) if rows else len(headers[3])
    w_start = max([len(r["start"]) for r in rows] + [len(headers[4])]) if rows else len(headers[4])

    pad_l = lambda s, w: s.ljust(w)
    pad_r = lambda s, w: s.rjust(w)
    line = f"+-{'-'*w_ma}-+-{'-'*w_chi}-+-{'-'*w_tm}-+-{'-'*w_end}-+-{'-'*w_start}-+"

    out = [line,
           "| "+pad_l(headers[0],w_ma)+" | "+pad_l(headers[1],w_chi)+" | "+pad_l(headers[2],w_tm)+
           " | "+pad_r(headers[3],w_end)+" | "+pad_r(headers[4],w_start)+" |",
           line]
    for r in rows:
        out.append("| "+pad_l(r['ma'],w_ma)+" | "+pad_l(r['chi'],w_chi)+" | "+pad_l(r['tm'],w_tm)+
                   " | "+pad_r(r['end'],w_end)+" | "+pad_r(r['start'],w_start)+" |")
    out.append(line)
    return "\n".join(out)

# ===== Làm sạch RAW header =====
_NOISE_LINE = re.compile(r"^[\W_]{1,4}$|^(he|y|m|per|\$f|~)$", re.IGNORECASE)
def clean_header_text(raw: str) -> str:
    if not raw: return ""
    lines = []
    for ln in raw.splitlines():
        s = ln.strip()
        if not s or len(s) <= 2 or _NOISE_LINE.match(s): continue
        # loại dòng giống bảng
        if "|" in s: continue
        if _NUM_GROUP.search(s): continue
        lines.append(ln)
    return "\n".join(lines).strip()

# ===== OCR ngoài bảng =====
def ocr_text_outside_table(orig_bgr: np.ndarray, table_bbox: Tuple[int,int,int,int]) -> str:
    x,y,w,h = table_bbox
    mask = np.zeros(orig_bgr.shape[:2], dtype=np.uint8)
    mask[y:y+h, x:x+w] = 255
    bg = orig_bgr.copy()
    bg[mask==255] = 255  # làm trắng vùng bảng
    raw_text = pytesseract.image_to_string(bg, lang=TESSERACT_LANG, config="--psm 6")
    return clean_header_text(raw_text)

# ===== Pipeline per-image =====
def process_img_to_text_and_table(bgr: np.ndarray) -> Tuple[str, str]:
    gray_bin = preprocess_image(bgr)
    bbox = find_table_bbox(gray_bin)
    table_text = ""; header_text = ""
    if bbox:
        x,y,w,h = bbox
        roi = bgr[y:y+h, x:x+w]
        rows = reflow_records(roi)
        table_text = format_table(rows) if rows else ""
        header_text = ocr_text_outside_table(bgr, bbox)
    else:
        rows = reflow_records(bgr)
        table_text = format_table(rows) if rows else ""
        header_text = ""
    return header_text, table_text

# ===== I/O =====
def save_outputs(header_text: str, table_text: str, out_dir: str, base: str):
    ensure_dir(out_dir)
    body = []
    if header_text:
        body.append(header_text); body.append("\n\n")
    if table_text:
        body.append(table_text)
    text = "".join(body)
    with open(os.path.join(out_dir, f"{base}.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    meta = {"text_sha1": _sha1(text), "has_table": bool(table_text), "has_header": bool(header_text)}
    with open(os.path.join(out_dir, f"{base}.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

# ===== Drivers =====
def process_image_file(img_path: str, out_dir: str):
    print(f"🖼️ Ảnh: {os.path.basename(img_path)}")
    bgr = cv2.imread(img_path)
    if bgr is None:
        print("⚠️ Không đọc được ảnh"); return
    header_text, table_text = process_img_to_text_and_table(bgr)
    base = os.path.splitext(os.path.basename(img_path))[0]
    save_outputs(header_text, table_text, out_dir, base)

def process_pdf_file(pdf_path: str, out_dir: str, start: int = 1, end: Optional[int] = None):
    print(f"📄 PDF: {os.path.basename(pdf_path)}")
    pages = convert_from_path(pdf_path, dpi=DPI, first_page=start, last_page=end)
    for i, pil_img in enumerate(pages, start=start):
        bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        header_text, table_text = process_img_to_text_and_table(bgr)
        base = f"{os.path.splitext(os.path.basename(pdf_path))[0]}_page{i}"
        save_outputs(header_text, table_text, out_dir, base)

def batch_process(input_dir: str, output_dir: str, start: int = 1, end: Optional[int] = None):
    for root, _, files in os.walk(input_dir):
        for file in sorted(files):
            path = os.path.join(root, file)
            ext  = os.path.splitext(file)[1].lower()
            if ext == ".pdf": process_pdf_file(path, output_dir, start, end)
            elif ext in EXT_IMAGE: process_image_file(path, output_dir)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser("BCTC OCR Pipeline")
    p.add_argument("--input",  default=INPUT_DIR_DEFAULT,  help="Thư mục chứa ảnh/PDF")
    p.add_argument("--output", default=OUTPUT_DIR_DEFAULT, help="Thư mục lưu kết quả")
    p.add_argument("--start", type=int, default=1, help="Trang bắt đầu của PDF")
    p.add_argument("--end",   type=int, default=None, help="Trang kết thúc của PDF")
    a = p.parse_args()
    batch_process(a.input, a.output, a.start, a.end)
    print("✅ Hoàn tất!")
