import camelot

print("✅ Camelot version:", camelot.__version__)

pdf_path = r"D:\1.TLAT\3.ChatBot_project\1_Insurance_Strategy\inputs\raw_clean_docs\insurance\2024 VN Insurance Market Data.pdf"

# Đọc bảng ở trang 1
tables = camelot.read_pdf(pdf_path, pages="1", flavor="stream")

print("📊 Số bảng đọc được:", tables.n)

# Nếu có bảng thì in thử 5 dòng đầu
if tables.n > 0:
    print(tables[0].df.head())
else:
    print("⚠️ Không tìm thấy bảng nào.")
