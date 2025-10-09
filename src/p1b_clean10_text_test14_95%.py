# -*- coding: utf-8 -*-
"""
p1b_clean10_text_test14.py
- Làm sạch OCR nâng cao (~98%)
- Phiên bản test14: gom số dính triệt để, bỏ số rác nhỏ
- Fix chữ OCR từ YAML (mặc định: src/p1b_clean10_ocr_fix.yaml)
- Xuất TXT, JSON, CSV
"""

import os, re, json, csv, argparse, yaml
from pathlib import Path

# ====== Load OCR_FIX từ YAML ======
def load_ocr_fix(yaml_file=None):
    paths_to_try = []
    if yaml_file:
        paths_to_try.append(Path(yaml_file))
    paths_to_try.append(Path(__file__).parent / "p1b_clean10_ocr_fix.yaml")
    paths_to_try.append(Path.cwd() / "p1b_clean10_ocr_fix.yaml")

    for p in paths_to_try:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                print(f"✅ Đang dùng OCR_FIX từ {p}")
                return yaml.safe_load(f)

    print("⚠ Không tìm thấy p1b_clean10_ocr_fix.yaml, dùng OCR_FIX rỗng")
    return {}

# ====== Hàm format số chuẩn VND ======
def format_num(num_str: str) -> str:
    try:
        return f"{int(num_str):,}".replace(",", ".")
    except:
        return ""

# ====== Hàm chuẩn hoá số ======
def clean_num(raw: str) -> str:
    if not raw:
        return ""

    # OCR nhầm chữ thành số
    raw = raw.replace("O", "0").replace("o", "0")
    raw = raw.replace("l", "1").replace("I", "1")
    raw = raw.replace("B", "8")

    # Gom số dính: bỏ hết dấu phẩy, chấm lạ
    digits = re.sub(r"[^\d]", "", raw)

    # Nếu quá ngắn (<6) → coi là số rác
    if len(digits) < 6:
        return ""

    return format_num(digits)

# ====== Hàm sửa text OCR ======
def clean_text(text: str, ocr_fix: dict) -> str:
    for wrong, right in ocr_fix.items():
        text = re.sub(rf"\b{wrong}\b", right, text, flags=re.IGNORECASE)
    text = text.replace("Wh", "").replace("£", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ====== Hàm parse 1 dòng ======
def parse_line(line: str, ocr_fix: dict):
    line = clean_text(line.strip(), ocr_fix)
    if not line:
        return None

    m = re.match(r"^(\d+[\.]?\d*)\s+(.*)", line)
    if not m:
        return None

    code = m.group(1)
    rest = m.group(2)

    nums = re.findall(r"[\d\.\,]+", rest)
    end_val, start_val = None, None
    if len(nums) >= 2:
        end_val, start_val = clean_num(nums[-2]), clean_num(nums[-1])
    elif len(nums) == 1:
        end_val = clean_num(nums[-1])

    # Loại số khỏi tên chỉ tiêu
    name = rest
    for num in nums[-2:]:
        name = name.replace(num, "")
    name = re.sub(r"\s+", " ", name).strip()

    return {
        "code": code,
        "name": name,
        "end_year": end_val if end_val else None,
        "start_year": start_val if start_val else None
    }

# ====== Hàm chính ======
def process_folder(in_dir: str, out_dir: str, yaml_file=None):
    os.makedirs(out_dir, exist_ok=True)
    ocr_fix = load_ocr_fix(yaml_file)

    for file in Path(in_dir).glob("*.txt"):
        with open(file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        clean_lines, data_file = [], []
        for line in lines:
            parsed = parse_line(line, ocr_fix)
            if parsed:
                data_file.append(parsed)
                clean_lines.append(
                    f"{parsed['code']} | {parsed['name']} | {parsed['end_year']} | {parsed['start_year']}"
                )
            else:
                clean_lines.append(clean_text(line, ocr_fix))

        # ghi txt
        out_txt = Path(out_dir) / file.name.replace(".txt", "_clean.txt")
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(clean_lines))

        # ghi json
        out_json = Path(out_dir) / file.name.replace(".txt", ".json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(data_file, f, ensure_ascii=False, indent=2)

        # ghi csv
        out_csv = Path(out_dir) / file.name.replace(".txt", ".csv")
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["code", "name", "end_year", "start_year"])
            writer.writeheader()
            writer.writerows(data_file)

    print(f"✅ Done. Cleaned files saved in {out_dir}")

# ====== CLI ======
def main():
    parser = argparse.ArgumentParser(description="Clean OCR text (general) → TXT, JSON, CSV (test14)")
    parser.add_argument("--in_dir", default=r"D:\1.TLAT\3. ChatBot_project\1_Insurance_Strategy - Copy\outputs\p1a_clean10_ocr_bctc")
    parser.add_argument("--out_dir", default=r"D:\1.TLAT\3. ChatBot_project\1_Insurance_Strategy - Copy\outputs\p1b_clean10_ocr_text_test14")
    parser.add_argument("--yaml_file", default=None, help="File YAML chứa từ điển OCR_FIX (mặc định: src/p1b_clean10_ocr_fix.yaml)")
    args = parser.parse_args()

    process_folder(args.in_dir, args.out_dir, args.yaml_file)

if __name__ == "__main__":
    main()
