import os
import glob
import argparse
import pandas as pd
import yaml
import unicodedata
import re
import pdfplumber  # thêm để đọc PDF

# ==== Helper ====
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def normalize_text(s):
    """Chuẩn hóa text: lower, bỏ dấu, bỏ khoảng trắng thừa"""
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")  # remove accents
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def to_number(val):
    """Chuyển giá trị OCR/Excel thành số"""
    if pd.isna(val):
        return None
    s = str(val).strip().replace(".", "").replace(",", "")
    if s.isdigit():
        return int(s)
    try:
        return float(s)
    except:
        return None


def make_unique_columns(cols):
    """Đảm bảo tên cột là duy nhất bằng cách thêm hậu tố .1, .2 nếu trùng"""
    seen = {}
    new_cols = []
    for c in cols:
        if c not in seen:
            seen[c] = 0
            new_cols.append(c)
        else:
            seen[c] += 1
            new_cols.append(f"{c}.{seen[c]}")
    return new_cols


def read_file(file_path):
    """Đọc Excel, CSV, hoặc PDF → trả về DataFrame"""
    ext = file_path.lower()
    try:
        if ext.endswith(".xlsx") or ext.endswith(".xls"):
            return pd.read_excel(file_path)
        elif ext.endswith(".csv"):
            return pd.read_csv(file_path)
        elif ext.endswith(".pdf"):
            tables = []
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    table = page.extract_table()
                    if table:
                        df = pd.DataFrame(table[1:], columns=table[0])
                        df.columns = make_unique_columns(df.columns)  # xử lý trùng tên
                        tables.append(df)
            if tables:
                return pd.concat(tables, ignore_index=True)
            else:
                print(f"⚠️ Không trích được bảng từ PDF: {file_path}")
                return None
        else:
            print(f"⚠️ Định dạng không hỗ trợ: {file_path}")
            return None
    except Exception as e:
        print(f"❌ Lỗi đọc file {file_path}: {e}")
        return None


def find_value(df, synonyms, target_keys, col_candidates):
    """Tìm giá trị trong DataFrame dựa vào synonyms"""
    for key in target_keys:
        if key not in synonyms:
            continue
        for kw in synonyms[key]:
            kw_norm = normalize_text(kw)
            # tìm hàng chứa từ khóa trong cột chỉ tiêu (thường là cột 1 hoặc 2)
            for col in df.columns[:2]:
                mask = df[col].astype(str).apply(lambda x: kw_norm in normalize_text(x))
                row = df[mask]
                if not row.empty:
                    for col_num in col_candidates:
                        if col_num in df.columns:
                            return to_number(row.iloc[0][col_num])
    return None

def analyze_file(file_path, synonyms, ratio_defs, output_dir, input_dir):
    df = read_file(file_path)
    if df is None or df.empty:
        return

    # Xác định cột số liệu dựa trên synonyms col_end + col_begin
    col_candidates = []
    for c in df.columns:
        c_norm = normalize_text(c)
        for kw in synonyms.get("col_end", []) + synonyms.get("col_begin", []):
            if normalize_text(kw) in c_norm:
                col_candidates.append(c)

    if not col_candidates:
        print(f"⚠️ Không tìm thấy cột số trong {file_path}. Cột có: {list(df.columns)}")
        return None

    results = {}
    for r in ratio_defs:
        num = find_value(df, synonyms, r["numerator"], col_candidates)
        den = find_value(df, synonyms, r["denominator"], col_candidates)
        if num and den:
            try:
                results[r["name"]] = round(num / den, 4)
            except ZeroDivisionError:
                results[r["name"]] = None

    if results:
        rel_path = os.path.relpath(file_path, input_dir)
        out_path = os.path.join(output_dir, rel_path)
        ensure_dir(os.path.dirname(out_path))
        out_file = re.sub(r"\.(xlsx|xls|csv|pdf)$", "_ratios.xlsx", out_path, flags=re.IGNORECASE)

        # Debug thêm
        print(">>> Results:", results)
        print(">>> Out file path:", out_file)

        pd.DataFrame([results]).to_excel(out_file, index=False)
        print(f"✅ Đã phân tích: {file_path} → {out_file}")

    
    
    else:
        print(f"⚠️ Không tính được tỷ lệ cho: {file_path}")
        print("   → Các cột có:", list(df.columns))
        print("   → 5 dòng đầu:")
        print(df.head().to_string())

def main(args):
    # Load cấu hình từ ratios.yaml
    with open("configs/ratios.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    synonyms = config.get("synonyms", {})
    ratio_defs = config.get("ratios", [])

    # Tìm Excel, CSV, PDF
    files = []
    for ext in ["*.xls*", "*.csv", "*.pdf"]:
        files.extend(glob.glob(os.path.join(args.input, "**", ext), recursive=True))

    print(f"🔍 Tìm thấy {len(files)} file trong {args.input}")

    for f in files:
        analyze_file(f, synonyms, ratio_defs, args.output, args.input)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ratio Engine cho phân tích tài chính")
    parser.add_argument("--input", required=True, help="Thư mục chứa file Excel/CSV/PDF đã clean hoặc file sạch")
    parser.add_argument("--output", required=True, help="Thư mục để lưu kết quả ratios")
    args = parser.parse_args()
    main(args)
