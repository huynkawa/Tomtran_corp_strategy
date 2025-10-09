import os

# 🧭 Thư mục gốc cần kiểm tra
BASE_DIR = r"inputs"

# 📏 Giới hạn an toàn (Windows ~260, nhưng ta để 240 cho chắc)
LIMIT = 240

report_lines = []
too_long = 0
total_files = 0

for root, dirs, files in os.walk(BASE_DIR):
    for f in files:
        total_files += 1
        path = os.path.join(root, f)
        abs_path = os.path.abspath(path)
        path_len = len(abs_path)
        mark = "⚠️" if path_len > LIMIT else "✅"
        line = f"{mark} {path_len:>4} | {abs_path}"
        report_lines.append(line)
        if path_len > LIMIT:
            too_long += 1

# 🧾 In kết quả tóm tắt ra console
print(f"\n🧮 Tổng số file kiểm tra: {total_files}")
print(f"⚠️ Số file vượt quá {LIMIT} ký tự: {too_long}")
if too_long > 0:
    print("→ Xem chi tiết trong 'long_path_report_inputs.txt'\n")
else:
    print("✅ Tất cả đường dẫn trong thư mục 'inputs/' đều an toàn.\n")

# 💾 Lưu báo cáo chi tiết
with open("long_path_report_inputs.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

print("📄 Đã lưu báo cáo: long_path_report_inputs.txt")
