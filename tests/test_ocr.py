import os
from pdf2image import convert_from_path
import pytesseract
from paddleocr import PaddleOCR
import numpy as np

# === Đường dẫn file PDF scan cần OCR ===
INPUT_FILE = r"D:\1.TLAT\3. ChatBot_project\1_Insurance_Strategy\inputs\UIC data\2024-UIC_Financial-Statements-V_signed.pdf"

# Chọn range trang cần OCR
START_PAGE = 9
END_PAGE = 10

# Thư mục output
OUT_DIR = "outputs/ocr_texts"
os.makedirs(OUT_DIR, exist_ok=True)

# Khởi tạo PaddleOCR
ocr_paddle = PaddleOCR(lang='vi')

# Kiểm tra file tồn tại
if not os.path.isfile(INPUT_FILE):
    raise FileNotFoundError(f"⚠️ Không tìm thấy file: {INPUT_FILE}")

# Convert PDF sang ảnh
pages = convert_from_path(INPUT_FILE, dpi=300)

for i in range(START_PAGE - 1, END_PAGE):  # -1 vì index = 0
    page_num = i + 1
    page = pages[i]

    print(f"\n==============================")
    print(f"📄 Trang {page_num}")
    print(f"==============================")

    # --- OCR với Tesseract ---
    text_tess = pytesseract.image_to_string(page, lang="vie")

    print("\n--- Tesseract ---")
    print(text_tess[:800])  # In 800 ký tự đầu tiên cho gọn

    # Lưu file Tesseract
    tess_path = os.path.join(OUT_DIR, f"page{page_num}_tesseract.txt")
    with open(tess_path, "w", encoding="utf-8") as f:
        f.write(text_tess)
    print(f"✅ Lưu Tesseract OCR: {tess_path}")

# --- OCR với PaddleOCR ---
import numpy as np
img_np = np.array(page)   # PIL → numpy
result = ocr_paddle.ocr(img_np)

paddle_text = ""

# PaddleOCR cũ (list of list)
if isinstance(result, list) and len(result) > 0 and isinstance(result[0], list):
    lines = result[0]
    print("\n--- PaddleOCR (kiểu list) ---")
    for line in lines[:15]:
        text = line[1][0]
        print(text)
        paddle_text += text + "\n"

# PaddleOCR mới (dict với "data")
elif isinstance(result, dict) and "data" in result:
    lines = result["data"]
    print("\n--- PaddleOCR (kiểu dict) ---")
    for line in lines[:15]:
        text = line.get("text", "")
        print(text)
        paddle_text += text + "\n"

else:
    print("⚠️ Không nhận diện được text")
    paddle_text = "⚠️ Không nhận diện được text"

# Lưu file PaddleOCR
paddle_path = os.path.join(OUT_DIR, f"page{page_num}_paddle.txt")
with open(paddle_path, "w", encoding="utf-8") as f:
    f.write(paddle_text)
print(f"✅ Lưu PaddleOCR: {paddle_path}")
