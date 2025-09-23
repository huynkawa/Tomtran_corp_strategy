# 📁 src/ocr_pipeline.py
import os
import re
import cv2
import glob
import shutil
import numpy as np
import pandas as pd
from pdf2image import convert_from_path
from paddleocr import PPStructure, save_structure_res
import pytesseract

# ==== Cấu hình ==== 
INPUT_FILE = None
INPUT_DIR = r"inputs/raw_scan"

RAW_DIR = "outputs/orc_raw_output"           # OCR thô (Excel + Text)
CLEAN_DIR = "outputs/clean_orc_raw_output"   # Sau khi làm sạch

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

# === Clean số cơ bản ===
def clean_number(val):
    if pd.isna(val):
        return val
    s = str(val).strip()

    # Giữ lại số, dấu . , -
    s = re.sub(r"[^\d,.\-]", "", s)

    # Nếu có cả , và . → giả định , là nghìn, . là thập phân
    if "," in s and "." in s:
        s = s.replace(",", "")
    # Nếu chỉ có , mà không có . → giả định , là nghìn
    elif "," in s and "." not in s:
        s = s.replace(",", "")
    # Nếu chỉ có . mà không có , → kiểm tra dạng nghìn
    elif "." in s and "," not in s:
        parts = s.split(".")
        if all(len(p) == 3 for p in parts[1:]):  # ví dụ 1.234.567
            s = "".join(parts)

    try:
        return float(s) if "." in s else int(s)
    except:
        return val

# === Clean text cơ bản ===
def clean_text(val):
    if not isinstance(val, str):
        return val
    text = val.strip()
    text = re.sub(r"\s+", " ", text)           # chuẩn hóa khoảng trắng
    text = re.sub(r"[^\w\s\-/.,]", "", text)   # bỏ ký tự lạ

    # Một số fix OCR phổ biến
    replacements = {
        "T otal": "Total",
        "T0tal": "Total",
        "2O24": "2024",
        "lién": "liên",
        "hiém": "hiểm",
    }
    for wrong, right in replacements.items():
        text = text.replace(wrong, right)

    return text

# === Clean Excel bằng pandas ===
def clean_excel(file_path, output_path):
    """Đọc và làm sạch file Excel rồi lưu vào CLEAN_DIR"""
    try:
        df = pd.read_excel(file_path)

        # Loại bỏ dòng/cột trống
        df.dropna(how="all", inplace=True)
        df.dropna(axis=1, how="all", inplace=True)

        # Chuẩn hóa tên cột
        df.columns = [clean_text(str(c)) for c in df.columns]

        # Làm sạch từng ô
        df_clean = df.copy()
        for col in df_clean.columns:
            df_clean[col] = df_clean[col].apply(
                lambda x: clean_number(x) if str(x).replace(".", "").replace(",", "").isdigit() else clean_text(x)
            )

        ensure_dir(os.path.dirname(output_path))
        df_clean.to_excel(output_path, index=False)
        print(f"✅ Cleaned Excel: {output_path}")
    except Exception as e:
        print(f"⚠️ Lỗi khi làm sạch {file_path}: {e}")

# === OCR 1 file PDF ===
def process_pdf(pdf_path, start_page=1, end_page=None, dpi=300):
    print(f"\n📂 Đang xử lý file: {pdf_path}")

    table_engine = PPStructure(show_log=True, use_gpu=False)
    pages = convert_from_path(pdf_path, dpi=dpi)
    total_pages = len(pages)
    if end_page is None or end_page > total_pages:
        end_page = total_pages
    print(f"📑 PDF có {total_pages} trang, xử lý từ {start_page} → {end_page}")

    # Giữ cấu trúc thư mục theo raw_scan
    rel_path = os.path.relpath(pdf_path, INPUT_DIR)
    rel_dir = os.path.dirname(rel_path)

    raw_subdir = os.path.join(RAW_DIR, rel_dir)
    clean_subdir = os.path.join(CLEAN_DIR, rel_dir)
    ensure_dir(raw_subdir)
    ensure_dir(clean_subdir)

    base_name = os.path.splitext(os.path.basename(pdf_path))[0]

    for i in range(start_page - 1, end_page):
        page_num = i + 1
        page = pages[i]
        img_np = np.array(page)
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # === OCR bảng với PaddleOCR ===
        result = table_engine(img_cv)
        try:
            save_structure_res(
                result,
                save_folder=raw_subdir,
                img_name=f"{base_name}_page{page_num}"
            )
        except Exception as e:
            print(f"⚠️ Lỗi save_structure_res: {e}")

        # === Move file Excel/Text từ folder con ra ngoài ===
        page_folder = os.path.join(raw_subdir, f"{base_name}_page{page_num}")
        excel_file_raw = os.path.join(raw_subdir, f"{base_name}_page{page_num}.xlsx")

        if os.path.isdir(page_folder):
            excel_files = glob.glob(os.path.join(page_folder, "*.xlsx"))
            txt_files = glob.glob(os.path.join(page_folder, "*.txt"))

            if excel_files:
                try:
                    shutil.move(excel_files[0], excel_file_raw)
                    print(f"📑 Xuất Excel RAW: {excel_file_raw}")

                    # === Clean và lưu sang CLEAN_DIR ===
                    clean_file = os.path.join(clean_subdir, f"{base_name}_page{page_num}.xlsx")
                    clean_excel(excel_file_raw, clean_file)

                except Exception as e:
                    print(f"⚠️ Lỗi move/read Excel: {e}")

            if txt_files:
                raw_text_file = os.path.join(raw_subdir, f"{base_name}_page{page_num}_ocr.txt")
                shutil.move(txt_files[0], raw_text_file)
                print(f"📝 Xuất Text RAW từ PaddleOCR: {raw_text_file}")

                # Copy sang CLEAN_DIR và làm sạch text
                clean_text_file = os.path.join(clean_subdir, f"{base_name}_page{page_num}_ocr.txt")
                ensure_dir(os.path.dirname(clean_text_file))
                with open(raw_text_file, "r", encoding="utf-8") as fr, open(clean_text_file, "w", encoding="utf-8") as fw:
                    for line in fr:
                        fw.write(clean_text(line) + "\n")
                print(f"📄 Copy Text sang CLEAN (đã clean cơ bản): {clean_text_file}")

            # Xóa folder con sau khi move xong
            shutil.rmtree(page_folder)

        # === Luôn OCR text bổ sung bằng Tesseract ===
        text_tess = pytesseract.image_to_string(img_cv, lang="eng+vie")
        text_file_raw = os.path.join(raw_subdir, f"{base_name}_page{page_num}_text.txt")
        with open(text_file_raw, "w", encoding="utf-8") as f:
            f.write(text_tess)
        print(f"📝 Xuất Text RAW từ Tesseract: {text_file_raw}")

        # Clean text và copy sang CLEAN_DIR
        text_file_clean = os.path.join(clean_subdir, f"{base_name}_page{page_num}_text.txt")
        ensure_dir(os.path.dirname(text_file_clean))
        with open(text_file_raw, "r", encoding="utf-8") as fr, open(text_file_clean, "w", encoding="utf-8") as fw:
            for line in fr:
                fw.write(clean_text(line) + "\n")
        print(f"📄 Copy Text sang CLEAN (đã clean cơ bản): {text_file_clean}")

# === Chạy toàn bộ OCR pipeline ===
def run_ocr_pipeline(start_page=1, end_page=None, dpi=300):
    # 🚨 Xóa thư mục RAW & CLEAN cũ trước khi chạy lại
    if os.path.exists(RAW_DIR):
        shutil.rmtree(RAW_DIR)
    if os.path.exists(CLEAN_DIR):
        shutil.rmtree(CLEAN_DIR)
    ensure_dir(RAW_DIR)
    ensure_dir(CLEAN_DIR)

    if INPUT_FILE:
        process_pdf(INPUT_FILE, start_page, end_page, dpi)
    else:
        pdf_files = glob.glob(os.path.join(INPUT_DIR, "**", "*.pdf"), recursive=True)
        if not pdf_files:
            print("⚠️ Không tìm thấy file PDF nào trong thư mục.")
            return
        for pdf in pdf_files:
            process_pdf(pdf, start_page, end_page, dpi)

if __name__ == "__main__":
    run_ocr_pipeline(start_page=1, end_page=None, dpi=300)
