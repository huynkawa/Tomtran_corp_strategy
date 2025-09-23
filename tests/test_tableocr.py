import os
import cv2
import numpy as np
from pdf2image import convert_from_path
from paddleocr import PPStructure, save_structure_res

# ==== Fix lỗi OneDNN trên CPU ====
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["CPU_NUM"] = "1"

# === Đường dẫn file PDF ===
INPUT_FILE = r"D:\1.TLAT\3. ChatBot_project\1_Insurance_Strategy\inputs\UIC data\2024-UIC_Financial-Statements-V_signed.pdf"

# Trang cần OCR bảng
START_PAGE = 7
END_PAGE = 9   # sẽ đọc từ 7 đến 9

# Thư mục output
OUT_DIR = "outputs/orc_table"
os.makedirs(OUT_DIR, exist_ok=True)

# Khởi tạo Table OCR
table_engine = PPStructure(show_log=True)

# Convert PDF sang ảnh
pages = convert_from_path(INPUT_FILE, dpi=400)   # tăng dpi cho nét hơn
print(f"📑 PDF có tổng cộng {len(pages)} trang")

# OCR từng trang
for i in range(START_PAGE - 1, END_PAGE):
    page_num = i + 1
    page = pages[i]

    print(f"\n==============================")
    print(f"📄 Trang {page_num} (Table OCR)")
    print(f"==============================")

    # PIL -> numpy (cv2)
    img_np = np.array(page)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # OCR bảng
    result = table_engine(img_cv)

    # Debug: xem kết quả OCR có gì
    print(f"🔍 Kết quả OCR raw (số phần tử): {len(result)}")
    for idx, item in enumerate(result):
        save_structure_res([item], save_folder=OUT_DIR, img_name=f"page{page_num}_item{idx}")


    # Lưu kết quả HTML/Excel nếu có bảng
    if result:
        save_structure_res(result, save_folder=OUT_DIR, img_name=f"page{page_num}")
        print(f"✅ Đã lưu: {OUT_DIR}\\page{page_num}.xlsx và {OUT_DIR}\\page{page_num}.html")
    else:
        print(f"⚠️ Không tìm thấy bảng nào trong trang {page_num}!")
