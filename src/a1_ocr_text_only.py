# -*- coding: utf-8 -*-
"""
a1_ocr_text_only.py — OCR và đọc văn bản đa định dạng (PDF, DOCX, EXCEL, CSV, IMAGE)

Mục tiêu:
---------
- Đọc tất cả các loại tài liệu (PDF, DOCX, XLSX, CSV, IMAGE, TXT)
- Không sinh ra bất kỳ file CSV/Excel nào.
- Chuyển toàn bộ nội dung (kể cả bảng scan, sơ đồ) sang text duy nhất.
- Xuất đúng 2 file/trang:
    <base>_page{n}_text.txt
    <base>_page{n}_meta.json
- Giữ cấu trúc thư mục mirror từ inputs sang outputs.
- Khi OCR ảnh hoặc PDF scan → tự động dùng thuật toán TSV reflow để tái cấu trúc bảng.

Đường dẫn mặc định:
-------------------
Input : D:\\1.TLAT\\3. ChatBot_project\\1_Insurance_Strategy\\inputs\\a_text_only_inputs
Output: D:\\1.TLAT\\3. ChatBot_project\\1_Insurance_Strategy\\outputs\\a_text_only_outputs

Yêu cầu thư viện:
-----------------
pip install pdf2image pillow opencv-python-headless numpy pytesseract python-docx pandas openpyxl tqdm
và cài đặt Tesseract (tesseract.exe có trong PATH hoặc đặt env TESSERACT_CMD)
"""

from __future__ import annotations
import os, re, glob, json, argparse, hashlib, shutil, time
from typing import Optional, Tuple, Dict, List
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
from pytesseract import Output as TessOutput
import pandas as pd
from tqdm import tqdm

try:
    import docx
except ImportError:
    docx = None


# =========================
# ⚙️ CẤU HÌNH CƠ BẢN
# =========================
INPUT_DIR_DEFAULT = r"D:\1.TLAT\3. ChatBot_project\1_Insurance_Strategy\inputs\a_text_only_inputs_test"
OUTPUT_DIR_DEFAULT = r"D:\1.TLAT\3. ChatBot_project\1_Insurance_Strategy\outputs\a_text_only_outputs"
OCR_LANG_DEFAULT = "vie+eng"
OCR_CFG_DEFAULT = "--psm 6 preserve_interword_spaces=1"
APPEND_MODE = False

TESSERACT_CMD = os.environ.get("TESSERACT_CMD", None)
if TESSERACT_CMD and os.path.isfile(TESSERACT_CMD):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


# =========================
# ⚙️ HÀM HỖ TRỢ CHUNG
# =========================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def clean_txt_chars(s: str) -> str:
    """Chuẩn hoá văn bản OCR: loại ký tự rác và khoảng trắng thừa"""
    if not s: return ""
    s = re.sub(r"[|¦•∙·]+", " ", s)
    s = re.sub(r"[^\S\r\n]{2,}", " ", s)
    s = re.sub(r"[^\x20-\x7E\u00A0-\uFFFF]", " ", s)
    return s.strip()

def _sha1_text(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8")).hexdigest()

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

def detect_language(text: str) -> str:
    if not text: return "vi"
    vi_marks = re.findall(r"[ăâêôơưđáàảãạéèẻẽẹíìỉĩịóòỏõọúùủũụýỳỷỹỵ]", text.lower())
    if len(vi_marks) >= 3: return "vi"
    if re.search(r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\b", text, re.IGNORECASE):
        return "en"
    return "vi"


# =========================
# ⚙️ TSV REFLOW từ p1a_clean10_ocr_bctc.py
# =========================
def reflow_lines_from_tsv_dict(data: Dict[str, List], y_tol: int = 4) -> str:
    """Ghép dòng TSV Tesseract thành văn bản mạch lạc hơn (phục vụ ảnh scan có bảng)."""
    n = len(data.get("text", []))
    groups: Dict[Tuple[int,int,int], List[int]] = {}
    for i in range(n):
        t = (data["text"][i] or "").strip()
        if not t: continue
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

    out_lines = []
    for ln in lines:
        s = ln["text"]
        # ép xuống dòng trước các mã số, mục lớn, số tiền
        s = re.sub(r"(?<!^)\s(?=\d{3}(?:\.\d+)?\b)", "\n", s)
        s = re.sub(r"(?<!^)\s(?=(?:I|II|III|IV|V)\.?\b)", "\n", s)
        s = re.sub(r"(?<!^)\s(?=\d{1,3}(?:[.,]\d{3}){2,}\b)", "\n", s)
        out_lines.extend([p.strip() for p in s.split("\n") if p.strip()])
    return "\n".join(out_lines)

def ocr_image_to_text_tsv(img_bgr, ocr_lang: str, ocr_cfg: str) -> str:
    """OCR ảnh bằng TSV reflow (đọc tốt hơn cho bảng / scan)."""
    try:
        cfg = (ocr_cfg or "").strip()
        cfg = re.sub(r"--psm\s+\d+", "", cfg)
        cfg = (cfg + " --psm 4 preserve_interword_spaces=1").strip()
        tsv = pytesseract.image_to_data(img_bgr, lang=ocr_lang, config=cfg, output_type=TessOutput.DICT)
        txt = reflow_lines_from_tsv_dict(tsv)
        return clean_txt_chars(txt)
    except Exception as e:
        print(f"⚠️ Lỗi OCR TSV: {e}")
        return ""


# =========================
# ⚙️ ĐỌC ẢNH & PDF SCAN
# =========================
def pdf_to_texts(pdf_path: str, dpi: int = 400,
                 ocr_lang: str = OCR_LANG_DEFAULT, ocr_cfg: str = OCR_CFG_DEFAULT,
                 start_page: Optional[int] = None, end_page: Optional[int] = None) -> List[Tuple[int, str]]:
    texts = []
    try:
        pages = convert_from_path(pdf_path, dpi=dpi)
        total = len(pages)
        s = start_page or 1
        e = end_page or total
        s = max(1, s); e = min(total, e)
        for i, page in enumerate(pages, start=1):
            if not (s <= i <= e):
                continue
            img = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
            txt = ocr_image_to_text_tsv(img, ocr_lang, ocr_cfg)
            texts.append((i, txt))
    except Exception as e:
        print(f"⚠️ Lỗi PDF {pdf_path}: {e}")
    return texts


# =========================
# ⚙️ ĐỌC FILE KHÔNG CẦN OCR (DOCX, EXCEL, CSV, TXT)
# =========================
def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def read_docx_file(path: str) -> str:
    if docx is None:
        raise ImportError("⚠️ Cần cài python-docx để đọc DOCX.")
    doc = docx.Document(path)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

def read_excel_or_csv(path: str, mode: str = "summary") -> str:
    ext = Path(path).suffix.lower()
    texts = []
    try:
        if ext == ".csv":
            df = pd.read_csv(path, dtype=str)
            dfs = {"CSV": df}
        else:
            xls = pd.ExcelFile(path, engine="openpyxl")
            dfs = {sheet: pd.read_excel(xls, sheet_name=sheet, dtype=str)
                   for sheet in xls.sheet_names}

        for sheet_name, df in dfs.items():
            if df.empty:
                continue
            df = df.fillna("").astype(str)
            if mode == "raw":
                headers = list(df.columns)
                header_line = " | ".join(headers)
                rows = [" | ".join(row.values) for _, row in df.iterrows()]
                sheet_text = f"\n\n========== SHEET: {sheet_name.upper()} ==========\n{header_line}\n" + "\n".join(rows)
            else:
                df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
                headers = list(df.columns)
                sheet_text = [f"\n========== SHEET: {sheet_name.upper()} =========="]
                sheet_text.append(f"Các cột gồm: {', '.join(headers)}.")
                for _, row in df.iterrows():
                    if not any(v.strip() for v in row.values):
                        continue
                    pairs = [f"{h}: {v}" for h, v in row.items() if v.strip()]
                    sheet_text.append("; ".join(pairs))
                sheet_text = "\n".join(sheet_text)
            texts.append(sheet_text.strip())

        return "\n\n".join(texts).strip()
    except Exception as e:
        print(f"⚠️ Lỗi đọc {path}: {e}")
        return ""


# =========================
# ⚙️ GHI OUTPUT
# =========================
def save_output_text_and_meta(text: str, meta: dict, out_txt: str, out_meta: str):
    ensure_dir(os.path.dirname(out_txt))
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(text)
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"📝 Saved: {os.path.basename(out_txt)}, {os.path.basename(out_meta)}")


# =========================
# ⚙️ XỬ LÝ FILE CHÍNH
# =========================
def process_file(file_path: str, input_root: str, output_root: str,
                 ocr_lang: str, ocr_cfg: str, dpi: int = 400,
                 start_page: Optional[int] = None, end_page: Optional[int] = None,
                 excel_mode: str = "summary"):

    rel_path = os.path.relpath(file_path, input_root)
    base = Path(file_path).stem
    out_dir = os.path.join(output_root, os.path.dirname(rel_path))
    ensure_dir(out_dir)

    ext = Path(file_path).suffix.lower()
    text_outputs = []

    if ext in [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"]:
        img = cv2.cvtColor(np.array(Image.open(file_path).convert("RGB")), cv2.COLOR_RGB2BGR)
        txt = ocr_image_to_text_tsv(img, ocr_lang, ocr_cfg)
        text_outputs = [(1, txt)]

    elif ext == ".pdf":
        all_pages = convert_from_path(file_path, dpi=dpi,
                                    first_page=start_page or 1,
                                    last_page=end_page or (start_page or 1))
        total = len(all_pages)
        print(f"📄 {os.path.basename(file_path)} → OCR {total} trang (từ {start_page or 1} đến {end_page or total})")

        for idx, page in enumerate(tqdm(all_pages, desc=f"OCR {os.path.basename(file_path)}", ncols=80)):
            i = (start_page or 1) + idx
            img = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
            t0 = time.perf_counter()
            txt = ocr_image_to_text_tsv(img, ocr_lang, ocr_cfg)
            t1 = time.perf_counter()
            print(f"🕓 Trang {i} hoàn tất ({t1 - t0:.1f}s)")
            text_outputs.append((i, txt))

    elif ext in [".doc", ".docx"]:
        try:
            txt = read_docx_file(file_path)
            text_outputs = [(1, txt)]
        except Exception as e:
            print(f"⚠️ Lỗi DOCX: {e}")

    elif ext in [".xls", ".xlsx", ".csv"]:
        txt = read_excel_or_csv(file_path, mode=excel_mode)
        text_outputs = [(1, txt)]

    elif ext == ".txt":
        txt = read_text_file(file_path)
        text_outputs = [(1, txt)]

    else:
        print(f"⚠️ Bỏ qua (định dạng không hỗ trợ): {file_path}")
        return

    # Ghi kết quả
    if len(text_outputs) > 1:
        combined_text = "\n\n".join(t for _, t in text_outputs if t.strip())
        page_range = f"{start_page or 1}-{end_page or (start_page or 1)}"
        out_txt = os.path.join(out_dir, f"{base}_page{page_range}_text.txt")
        out_meta = os.path.join(out_dir, f"{base}_page{page_range}_meta.json")
        meta = {
            "file": base,
            "page_range": page_range,
            "page_count": len(text_outputs),
            "source_path": os.path.abspath(file_path),
            "language": detect_language(combined_text),
            "ocr_lang": ocr_lang,
            "ocr_cfg": ocr_cfg,
            "text_sha1": _sha1_text(combined_text),
        }
        save_output_text_and_meta(combined_text, meta, out_txt, out_meta)
    else:
        for page_no, txt in text_outputs:
            out_txt = os.path.join(out_dir, f"{base}_page{page_no}_text.txt")
            out_meta = os.path.join(out_dir, f"{base}_page{page_no}_meta.json")
            meta = {
                "file": base,
                "page": page_no,
                "source_path": os.path.abspath(file_path),
                "language": detect_language(txt),
                "ocr_lang": ocr_lang,
                "ocr_cfg": ocr_cfg,
                "text_sha1": _sha1_text(txt),
            }
            save_output_text_and_meta(txt, meta, out_txt, out_meta)


# =========================
# ⚙️ MAIN ENTRYPOINT
# =========================
def main():
    parser = argparse.ArgumentParser("A1 — OCR đa định dạng (Text only, có TSV reflow cho bảng)")
    parser.add_argument("--input", type=str, default=INPUT_DIR_DEFAULT, help="Thư mục input")
    parser.add_argument("--out", type=str, default=OUTPUT_DIR_DEFAULT, help="Thư mục output")
    parser.add_argument("--ocr-lang", type=str, default=OCR_LANG_DEFAULT)
    parser.add_argument("--ocr-cfg", type=str, default=OCR_CFG_DEFAULT)
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--clean", choices=["y","a","n","ask"], default="ask")
    parser.add_argument("--start", type=int, default=None, help="Trang bắt đầu (chỉ áp dụng cho PDF)")
    parser.add_argument("--end", type=int, default=None, help="Trang kết thúc (chỉ áp dụng cho PDF)")
    parser.add_argument("--excel-mode", choices=["raw", "summary"], default="summary",
                        help="Cách đọc Excel: raw=giữ nguyên, summary=làm sạch để vector store")

    args = parser.parse_args()
    START_PAGE, END_PAGE = args.start, args.end

    if os.path.exists(args.out):
        choice = args.clean
        if choice == "ask":
            choice = input(f"⚠️ Output '{args.out}' đã tồn tại. y=xoá, a=append, n=bỏ qua: ").strip().lower()
        if choice == "y":
            shutil.rmtree(args.out, ignore_errors=True)
            print(f"🗑️ Đã xoá {args.out}")
        elif choice == "a":
            global APPEND_MODE
            APPEND_MODE = True
            print(f"➕ Giữ {args.out}, chỉ ghi file mới.")
        elif choice == "n":
            print("⏭️ Bỏ qua toàn bộ."); return
        else:
            print("❌ Lựa chọn không hợp lệ."); return

    ensure_dir(args.out)
    files = glob.glob(os.path.join(args.input, "**", "*.*"), recursive=True)
    if not files:
        print("⚠️ Không tìm thấy file nào trong input."); return

    print(f"📂 Input: {args.input}")
    print(f"📦 Output: {args.out}")
    print(f"🧮 Tổng số file: {len(files)}")
    print(f"🧭 Giới hạn trang PDF: {START_PAGE or 1} → {END_PAGE or 'tất cả'}")
    has_excel = any(f.lower().endswith((".xls", ".xlsx", ".csv")) for f in files)
    if has_excel:
        print(f"📊 Chế độ đọc Excel: {args.excel_mode}")

    for f in files:
        process_file(f, args.input, args.out, args.ocr_lang, args.ocr_cfg,
                     dpi=args.dpi, start_page=START_PAGE, end_page=END_PAGE,
                     excel_mode=args.excel_mode)

    print("\n✅ Hoàn tất OCR. Kiểm tra *_text.txt và *_meta.json trong thư mục output.")


if __name__ == "__main__":
    main()
