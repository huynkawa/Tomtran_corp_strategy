# 📁 src/ocr_pipeline.py
import os
import cv2
import numpy as np
import pandas as pd
from pdf2image import convert_from_path
from paddleocr import PPStructure, save_structure_res
import pytesseract
import glob
import shutil

# ==== Cấu hình ==== 
INPUT_FILE = None
INPUT_DIR = r"inputs/raw_scan"
TEMP_DIR = "outputs/orc_raw_output"          # OCR thô (Excel + Text)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

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

    temp_subdir = os.path.join(TEMP_DIR, rel_dir)
    ensure_dir(temp_subdir)
   

    base_name = os.path.splitext(os.path.basename(pdf_path))[0]

    for i in range(start_page - 1, end_page):
        page_num = i + 1
        page = pages[i]
        img_np = np.array(page)
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # === OCR bảng với PaddleOCR ===
        result = table_engine(img_cv)
        try:
            save_structure_res(result, save_folder=temp_subdir, img_name=f"{base_name}_page{page_num}")
        except Exception as e:
            print(f"⚠️ Lỗi save_structure_res: {e}")

        # === Move file Excel/Text từ folder con ra ngoài ===
        page_folder = os.path.join(temp_subdir, f"{base_name}_page{page_num}")
        excel_file_raw = os.path.join(temp_subdir, f"{base_name}_page{page_num}.xlsx")
        df = None

        if os.path.isdir(page_folder):
            excel_files = glob.glob(os.path.join(page_folder, "*.xlsx"))
            txt_files = glob.glob(os.path.join(page_folder, "*.txt"))

            if excel_files:
                try:
                    shutil.move(excel_files[0], excel_file_raw)
                    print(f"📑 Xuất Excel RAW: {excel_file_raw}")
                    df = pd.read_excel(excel_file_raw)
                except Exception as e:
                    print(f"⚠️ Lỗi move/read Excel: {e}")

            if txt_files:
                raw_text_file = os.path.join(temp_subdir, f"{base_name}_page{page_num}_ocr.txt")
                shutil.move(txt_files[0], raw_text_file)
                print(f"📝 Xuất Text RAW từ PaddleOCR: {raw_text_file}")

            # Chỉ xóa folder con sau khi move xong
            shutil.rmtree(page_folder)

        # === Luôn OCR text bổ sung bằng Tesseract ===
        text_tess = pytesseract.image_to_string(img_cv, lang="eng+vie")
        text_file_raw = os.path.join(temp_subdir, f"{base_name}_page{page_num}_text.txt")
        with open(text_file_raw, "w", encoding="utf-8") as f:
            f.write(text_tess)
        print(f"📝 Xuất Text RAW từ Tesseract: {text_file_raw}")


def run_ocr_pipeline(start_page=1, end_page=None, dpi=300):

    # Xóa thư mục RAW cũ trước khi chạy lại
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    ensure_dir(TEMP_DIR)

    if INPUT_FILE:
        process_pdf(INPUT_FILE, start_page, end_page, dpi)
    else:
        pdf_files = glob.glob(os.path.join(INPUT_DIR, "**", "*.pdf"), recursive=True)
        if not pdf_files:
            print("⚠️ Không tìm thấy file PDF nào trong thư mục.")
            return
        for pdf in pdf_files:
            process_pdf(pdf, start_page, end_page, dpi)



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
    run_ocr_pipeline(start_page=5, end_page=17, dpi=300)
