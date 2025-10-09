"""
p1_clean10_version3_92_fix.py
Cải tiến pipeline OCR trích xuất BCTC:
- Giữ cấu trúc 5 cột: [Mã số | Chỉ tiêu | Thuyết minh | Số cuối năm | Số đầu năm]
- Giảm lỗi mất giá trị, sai cột, gộp dòng
- Hỗ trợ debug trực quan bằng đường ranh bảng và cột
"""

import os, re, cv2, json, unicodedata, hashlib
import numpy as np
import pytesseract
from pytesseract import Output
from pdf2image import convert_from_path
from typing import List, Tuple, Optional, Dict

# ===== CONFIG =====
TESSERACT_LANG = "vie+eng"
TESSERACT_CFG  = "--psm 6 -c preserve_interword_spaces=1"
DPI = 500
EXT_IMAGE = [".jpg", ".jpeg", ".png"]
INPUT_DIR_DEFAULT  = r"outputs/p1_prelight_ocr_bctc"
OUTPUT_DIR_DEFAULT = r"outputs/p1_clean10_version4"

# ===== UTILS =====
def ensure_dir(p): os.makedirs(p, exist_ok=True)
def _sha1(s: str) -> str: return hashlib.sha1(s.encode("utf-8")).hexdigest()

def preprocess_image(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    th   = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY,21,8)
    return th

def strip_vn_accents(s: str) -> str:
    s = (s or "").replace("Đ","D").replace("đ","d")
    s = unicodedata.normalize("NFD", s)
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn")

def norm_vn(s: str) -> str:
    s = strip_vn_accents(s).lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+"," ",s).strip()

# ===== CLEAN NUMBER =====
def clean_number(s: str) -> str:
    if not s: return ""
    s = (s or "").replace("O","0").replace("o","0").replace("Ô","0").replace("U","0")
    s = s.replace(",", "").replace(" ", "")
    s = re.sub(r"[^0-9\.]", "", s)
    # Giữ dấu . trong nhóm nghìn
    s = re.sub(r"(?<=\d)\.(?=\d{3}\b)", "", s)
    s = re.sub(r"\.{2,}", ".", s)
    return s.strip(".")

# ===== DETECT TABLE =====
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
        if w*h > 0.03*W*H and w > 0.3*W and h > 0.2*H:
            return x,y,w,h
    return None

# ===== DETECT COLUMN SPLIT =====
NUM_TOKEN = re.compile(r"[0-9][0-9\.,]{2,}")

def find_columns_by_headers(words: List[Dict], page_h: int) -> Optional[Tuple[int,int]]:
    top_cut = int(page_h * 0.6)
    head = [w for w in words if w['top'] < top_cut]
    if not head: return None
    by_line = {}
    for w in head: by_line.setdefault(w['line_num'], []).append(w)
    end_x = start_x = None
    for toks in by_line.values():
        toks.sort(key=lambda x: x['left'])
        text = " ".join(t['text'] for t in toks)
        n = norm_vn(text)
        if "so cuoi nam" in n: end_x  = min(t['left'] for t in toks)
        if "so dau nam"  in n: start_x = min(t['left'] for t in toks)
    if end_x and start_x:
        split1 = max(min(end_x, start_x) - 30, 180)
        split2 = (end_x + start_x) // 2
        if split2 - split1 < 250:
            split2 += 150
        return split1, max(split2, split1+120)
    return None

def find_columns_fallback_by_numbers(words: List[Dict], W: int) -> Tuple[int,int]:
    xs = [w['left'] + w['width']/2 for w in words if NUM_TOKEN.search(w['text']) and (w['left']+w['width']/2)>W*0.45]
    if len(xs)>=6:
        xs = np.array(xs)
        c1,c2 = np.percentile(xs,[35,70])
        split1 = int(c1-50); split2=int(c2)
    else:
        split1, split2 = int(W*0.58), int(W*0.78)
    return max(180,split1), max(split1+120,split2)

# ===== PARSE LEFT SIDE =====
_re_code = re.compile(r"^\s*[\+\|\.\-: ]{0,8}(\d{3})\b")
_re_note = re.compile(r"(?:^|\s)(\d{1,2}(?:\.\d)?)\s*$")

def parse_left_side(text: str) -> Tuple[str, str, str]:
    s = (text or "").strip()
    s = re.sub(r"^[\+\|\.\-: ]+", "", s)
    ma = chi = tm = ""
    m = _re_code.match(s)
    if m:
        ma = m.group(1); s = s[m.end():].strip(" .-:|+")
    m = _re_note.search(s)
    if m:
        tm = m.group(1); s = s[:m.start()].strip(" .-:|+")
    chi = s
    return ma, chi, tm

# ===== NUMBER HANDLING =====
_NUM_GROUP = re.compile(r"\d{1,3}(?:\.\d{3})+")

def is_note_token(text: str) -> bool:
    t = (text or "").strip()
    return bool(re.fullmatch(r"\d{1,2}(?:\.\d)?", t))

def tokens_to_number(tokens: List[Dict]) -> str:
    s = " ".join(t["text"] for t in tokens)
    cands = _NUM_GROUP.findall(s)
    if cands:
        return clean_number(max(cands, key=lambda x: len(x)))
    return clean_number(s)

# ===== REFLOW RECORDS =====
def reflow_records(img_bgr: np.ndarray, debug_path: Optional[str]=None) -> List[Dict]:
    data = pytesseract.image_to_data(img_bgr, lang=TESSERACT_LANG, config=TESSERACT_CFG, output_type=Output.DICT)
    words=[]
    for i,t in enumerate(data['text']):
        txt=(t or "").strip()
        if not txt: continue
        try: conf=float(data['conf'][i])
        except: conf=0
        if conf<0: continue
        words.append({
            "text":txt,"left":int(data['left'][i]),"top":int(data['top'][i]),
            "width":int(data['width'][i]),"line_num":int(data['line_num'][i]),
            "par_num":int(data['par_num'][i]),"block_num":int(data['block_num'][i])
        })
    if not words: return []

    H,W = img_bgr.shape[:2]
    split1,split2 = (find_columns_by_headers(words,H) or find_columns_fallback_by_numbers(words,W))

    # Vẽ debug line
    if debug_path:
        dbg = img_bgr.copy()
        cv2.line(dbg,(split1,0),(split1,H),(0,255,0),2)
        cv2.line(dbg,(split2,0),(split2,H),(0,0,255),2)
        cv2.imwrite(debug_path,dbg)

    groups={}
    for w in words: groups.setdefault((w['block_num'],w['par_num'],w['line_num']),[]).append(w)
    raw=[]
    for toks in groups.values():
        toks.sort(key=lambda x:(x['top'],x['left'])); raw.append(toks)
    raw.sort(key=lambda arr: arr[0]['top'])

    merged=[]
    for arr in raw:
        if not merged: merged.append(arr); continue
        last=merged[-1]
        if abs(arr[0]['top'] - last[-1]['top']) <= 10:  # tăng tolerance
            last.extend(arr); last.sort(key=lambda t:t['left'])
        else:
            merged.append(arr)

    rows=[]
    for toks in merged:
        toks.sort(key=lambda t:t['left'])
        left_tokens=[]; end_tokens=[]; start_tokens=[]
        for t in toks:
            xmid=t['left']+t['width']/2
            if xmid < split1: left_tokens.append(t)
            elif xmid < split2: end_tokens.append(t)
            else: start_tokens.append(t)
        # Lọc bỏ thuyết minh khỏi cột tiền
        end_tokens=[t for t in end_tokens if not is_note_token(t['text'])]
        start_tokens=[t for t in start_tokens if not is_note_token(t['text'])]
        left_text=" ".join(t['text'] for t in left_tokens)
        ma,chi,tm=parse_left_side(left_text)
        end_num=tokens_to_number(end_tokens)
        start_num=tokens_to_number(start_tokens)
        if ma or chi or tm or end_num or start_num:
            rows.append({"ma":ma,"chi":chi,"tm":tm,"end":end_num,"start":start_num})

    # Gộp dòng ngắn nối tiếp
    fused=[]
    for r in rows:
        if fused and not r["ma"] and not r["tm"] and not r["end"] and not r["start"] and len(r["chi"])<=28:
            fused[-1]["chi"]=(fused[-1]["chi"]+" "+r["chi"]).strip()
        else:
            fused.append(r)
    return fused

# ===== FORMAT OUTPUT =====
def format_table(rows: List[Dict]) -> str:
    headers=["Mã số","Chỉ tiêu","Thuyết minh","Số cuối năm","Số đầu năm"]
    out=[f"+{'-'*6}+{'-'*45}+{'-'*10}+{'-'*20}+{'-'*20}+"]
    out.append(f"| {' | '.join(headers)} |")
    out.append(f"+{'-'*6}+{'-'*45}+{'-'*10}+{'-'*20}+{'-'*20}+")
    for r in rows:
        out.append(f"| {r['ma']:<6}| {r['chi'][:45]:<45}| {r['tm']:<10}| {r['end']:<20}| {r['start']:<20}|")
    out.append(f"+{'-'*6}+{'-'*45}+{'-'*10}+{'-'*20}+{'-'*20}+")
    return "\n".join(out)

# ===== PIPELINE =====
def process_img_to_text_and_table(bgr: np.ndarray, debug_dir: Optional[str]=None) -> Tuple[str, str]:
    gray_bin = preprocess_image(bgr)
    bbox=find_table_bbox(gray_bin)
    header_text=""; table_text=""
    if bbox:
        x,y,w,h=bbox
        roi=bgr[y:y+h,x:x+w]
        debug_path=None
        if debug_dir: 
            ensure_dir(debug_dir)
            debug_path=os.path.join(debug_dir,"debug_bbox.png")
        rows=reflow_records(roi,debug_path)
        table_text=format_table(rows) if rows else ""
    else:
        rows=reflow_records(bgr)
        table_text=format_table(rows) if rows else ""
    return header_text, table_text

def process_image_file(img_path:str,out_dir:str,debug_dir:str=None):
    print(f"🖼️ {os.path.basename(img_path)}")
    bgr=cv2.imread(img_path)
    if bgr is None: return
    head,table=process_img_to_text_and_table(bgr,debug_dir)
    base=os.path.splitext(os.path.basename(img_path))[0]
    ensure_dir(out_dir)
    with open(os.path.join(out_dir,f"{base}.txt"),"w",encoding="utf-8") as f:
        f.write(table)

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--input",default=INPUT_DIR_DEFAULT)
    p.add_argument("--output",default=OUTPUT_DIR_DEFAULT)
    p.add_argument("--debug",default="debug_outputs")
    a=p.parse_args()
    ensure_dir(a.output)
    ensure_dir(a.debug)
    for root,_,files in os.walk(a.input):
        for file in sorted(files):
            if file.lower().endswith(tuple(EXT_IMAGE)):
                process_image_file(os.path.join(root,file),a.output,a.debug)
    print("✅ Hoàn tất v3.92_fix!")
