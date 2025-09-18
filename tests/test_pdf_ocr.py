from src.ingest import ocr_pdf

file_path = "inputs/financial_data/2024-UIC_Financial-Statements-V_signed.pdf"
text = ocr_pdf(file_path)

if text.strip():
    print(f"OCR thành công, đọc được {len(text)} ký tự")
    print("--- Trích 500 ký tự đầu ---")
    print(text[:500])
else:
    print("⚠️ OCR thất bại hoặc không có chữ nào")
