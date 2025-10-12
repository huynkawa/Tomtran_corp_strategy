# -*- coding: utf-8 -*-
"""
a2_ocr_mixed_text_table_GPT.py
---------------------------------------
Đọc và xử lý tài liệu hỗn hợp (văn bản + bảng/ảnh):
- Tự nhận dạng loại file: DOCX, PDF, TXT, hình scan.
- OCR khi cần (không lưu .png, xử lý trong RAM).
- GPT chỉ chỉnh sửa phần bảng hoặc ảnh.
- Xuất file text + meta JSON (metadata mở rộng cho vector store).

Author: TOMTRAN (pipeline mixed document version)
"""

import os, re, io, json, hashlib, argparse
from pathlib import Path
from typing import List, Dict, Optional

import fitz  # PyMuPDF
import pdfplumber
from PIL import Image
import pytesseract
from docx import Document
from tqdm import tqdm
from langdetect import detect

import src.env  # nạp OPENAI_API_KEY từ .env.active
from src.gpt_enhancer import enhance_with_gpt

import yaml


# ===== GPT enhancer (an toàn có fallback) =====
try:
    # nếu chạy kiểu: python -m src.a1_ocr_mixed_text_table_GPT
    from src.gpt_enhancer import enhance_with_gpt as _enhance_with_gpt
except Exception:
    try:
        # nếu file đang trong cùng package
        from .gpt_enhancer import enhance_with_gpt as _enhance_with_gpt
    except Exception:
        # fallback: không dùng GPT, trả nguyên văn để không bị NameError
        def _enhance_with_gpt(text, meta=None, image=None, **kwargs):
            return text

# ========== Cấu hình mặc định ==========
# ========== Cấu hình mặc định (linh hoạt & portable) ==========

import os
from pathlib import Path

# Xác định thư mục gốc của project (nơi chứa thư mục src/)
BASE_DIR = Path(__file__).resolve().parent.parent


# ========== Cấu hình mặc định (bản tuyệt đối, cố định cho máy D:\) ==========

SAVE_PNG_DEBUG = False
OCR_LANG = "vie+eng"
OCR_CFG = "--psm 6 preserve_interword_spaces=1"

# Đường dẫn tuyệt đối (đảm bảo nhất cho pipeline cục bộ)
INPUT_DIR  = r"D:\1.TLAT\3. ChatBot_project\1_Insurance_Strategy\inputs\c_financial_reports_test"
OUTPUT_DIR = r"D:\1.TLAT\3. ChatBot_project\1_Insurance_Strategy\outputs\a1_ocr_mixed_text_table_GPT_test so sanh p1a"
YAML_RULE_PATH = r"D:\1.TLAT\3.ChatBot_project\1_Insurance_Strategy\configs\a1_text_only_rules.yaml"

# ========== Tiện ích chung ==========
def sha1_of_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def detect_language_safe(text: str) -> str:
    try:
        return detect(text)
    except Exception:
        return "unknown"


def load_yaml_rules(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def match_yaml_meta(file_name: str, rules: dict) -> dict:
    """Tìm meta mặc định dựa theo regex trong YAML."""
    if not rules or "files" not in rules:
        return {}
    for rule in rules["files"]:
        if re.search(rule.get("match", ""), file_name, flags=re.I):
            meta = rule.copy()
            meta.pop("match", None)
            return {**rules.get("defaults", {}), **meta}
    return rules.get("defaults", {})


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


# ========== Xử lý DOCX ==========
def read_docx_text_and_tables(file_path: Path) -> List[Dict[str, str]]:
    """Đọc nội dung từ DOCX, tách giữa đoạn văn và bảng."""
    doc = Document(file_path)
    results = []

    for block in doc.element.body:
        if block.tag.endswith("tbl"):  # Bảng
            rows = []
            for r in block.findall(".//w:tr", block.nsmap):
                cells = [c.text for c in r.findall(".//w:t", block.nsmap)]
                rows.append(" | ".join(cells))
            text_block = "\n".join(rows)
            results.append({"type": "table", "content": text_block})
        else:  # Đoạn văn
            text = "".join(t.text for t in block.findall(".//w:t", block.nsmap)).strip()
            if text:
                results.append({"type": "paragraph", "content": text})
    return results


# ========== Xử lý PDF ==========
def read_pdf_text_and_images(file_path: Path) -> List[Dict[str, str]]:
    """Đọc PDF: text + ảnh (OCR ảnh khi cần)."""
    results = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                if text.strip():
                    results.append({"type": "paragraph", "content": text})
                # Nếu có ảnh: OCR vùng ảnh
                for img in page.images:
                    x0, y0, x1, y1 = img["x0"], img["y0"], img["x1"], img["y1"]
                    bbox = (x0, y0, x1, y1)
                    cropped = page.within_bbox(bbox).to_image(resolution=300).original
                    img_pil = Image.open(io.BytesIO(cropped))
                    txt = pytesseract.image_to_string(
                        img_pil, lang=OCR_LANG, config=OCR_CFG
                    )
                    if txt.strip():
                        results.append({"type": "table", "content": txt})
    except Exception as e:
        print(f"⚠️ PDF đọc lỗi {file_path}: {e}")
    return results


# ========== Xử lý hình ảnh ==========
def ocr_image_to_text(file_path: Path) -> str:
    """OCR ảnh sang text (OCR trong RAM)."""
    img = Image.open(file_path)
    return pytesseract.image_to_string(img, lang=OCR_LANG, config=OCR_CFG)


# ========== Xử lý TXT ==========
def read_txt(file_path: Path) -> str:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

# ========== Xử lý Excel (multi-sheet) ==========
import pandas as pd

def read_excel_as_text(file_path: Path) -> List[Dict[str, str]]:
    """
    Đọc file Excel (.xlsx, .xls) gồm nhiều sheet.
    Mỗi sheet trả về 1 block {"sheet": <tên>, "content": <text bảng>}.
    """
    results = []
    try:
        sheets = pd.read_excel(file_path, sheet_name=None, header=None)
        for sheet_name, df in sheets.items():
            df = df.fillna("")
            rows = []
            for row in df.values.tolist():
                row_text = " | ".join(str(cell).strip() for cell in row)
                rows.append(row_text)
            table_text = "\n".join(rows)
            results.append({
                "sheet": sheet_name,
                "content": table_text
            })
    except Exception as e:
        print(f"⚠️ Excel đọc lỗi {file_path}: {e}")
    return results





# ========== Xử lý chính ==========
def process_file(file_path: Path, yaml_rules: dict):
    file_name = file_path.stem
    meta = {
        "file": file_name,
        "source_path": str(file_path.resolve()),
        "ocr_lang": OCR_LANG,
        "ocr_cfg": OCR_CFG,
    }
    meta.update(match_yaml_meta(file_name, yaml_rules))

    output_dir = Path(OUTPUT_DIR)
    ensure_dir(output_dir)

    ext = file_path.suffix.lower()
    has_table = False
    gpt_applied = False
    combined_text = ""

    try:
        if ext in [".docx"]:
            blocks = read_docx_text_and_tables(file_path)
            for block in blocks:
                if block["type"] == "table":
                    has_table = True
                    gpt_applied = True
                    new_txt = _enhance_with_gpt(block["content"], meta, None)
                    combined_text += "\n" + new_txt
                else:
                    combined_text += "\n" + block["content"]

        elif ext in [".pdf"]:
            blocks = read_pdf_text_and_images(file_path)
            for block in blocks:
                if block["type"] == "table":
                    has_table = True
                    gpt_applied = True
                    new_txt = _enhance_with_gpt(block["content"], meta, None)
                    combined_text += "\n" + new_txt
                else:
                    combined_text += "\n" + block["content"]

        elif ext in [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"]:
            has_table = True
            gpt_applied = True
            txt = ocr_image_to_text(file_path)
            combined_text += "\n" + _enhance_with_gpt(txt, meta, file_path)

        elif ext in [".txt", ".csv"]:
            txt = read_txt(file_path)
            combined_text += "\n" + txt



        elif ext in [".xlsx", ".xls"]:
            excel_blocks = read_excel_as_text(file_path)
            if excel_blocks:
                has_table = True
                gpt_applied = True
                all_text = "\n\n".join(
                    [f"--- Sheet: {b['sheet']} ---\n{b['content']}" for b in excel_blocks]
                )
                enhanced_txt = _enhance_with_gpt(all_text, meta, None)
                combined_text += "\n" + enhanced_txt

        else:
            print(f"⚠️ Không hỗ trợ định dạng {ext}")
            return

        # enrich meta
        lang = detect_language_safe(combined_text)
        meta.update(
            {
                "language": lang,
                "has_table": has_table,
                "gpt_applied": gpt_applied,
                "text_sha1": sha1_of_text(combined_text),
            }
        )

        # Ghi file output
        txt_out = output_dir / f"{file_name}_text.txt"
        meta_out = output_dir / f"{file_name}_meta.json"
        with open(txt_out, "w", encoding="utf-8") as f:
            f.write(combined_text.strip())
        with open(meta_out, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print(f"✅ Saved: {txt_out.name}, {meta_out.name}")

    except Exception as e:
        print(f"❌ Lỗi xử lý {file_path}: {e}")


# ========== CLI ==========
def main():
    parser = argparse.ArgumentParser(
        description="OCR + GPT cho tài liệu hỗn hợp (text + bảng + ảnh)"
    )
    parser.add_argument("--clean", choices=["y", "n"], default="n")
    parser.add_argument("--start", type=int, default=1, help="Số thứ tự file bắt đầu (tính từ 1)")
    parser.add_argument("--end", type=int, default=None, help="Số thứ tự file kết thúc (nếu không có thì chạy đến hết)")
    args = parser.parse_args()

    yaml_rules = load_yaml_rules(YAML_RULE_PATH)
    files = sorted(list(Path(INPUT_DIR).rglob("*.*")))

    # Lọc theo start–end
    selected = files[args.start - 1 : args.end] if args.end else files[args.start - 1 :]

    print("=== CẤU HÌNH HIỆN TẠI ===")
    print(f"📂 INPUT_DIR   : {INPUT_DIR}")
    print(f"📦 OUTPUT_DIR  : {OUTPUT_DIR}")
    print(f"⚙️ YAML_RULE   : {YAML_RULE_PATH}")
    print(f"🧠 Tổng số file: {len(files)} | Sẽ xử lý: {len(selected)} (từ {args.start} đến {args.end or len(files)})")
    print("==========================")

    for file_path in tqdm(selected, desc="Processing"):
        process_file(file_path, yaml_rules)

    print("🎯 Hoàn tất A2. Kiểm tra thư mục output để xem kết quả.")


if __name__ == "__main__":
    main()
