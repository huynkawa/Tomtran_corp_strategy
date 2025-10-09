# 📁 src/clean_raw_output.py
import os
import re
import glob
import shutil
import argparse
import pandas as pd

RAW_DIR = r"outputs/orc_raw_output"           # dữ liệu OCR thô
CLEAN_DIR = r"outputs/clean_orc_raw_output"   # dữ liệu sau khi clean bước 1

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

NUM_PATTERN = re.compile(r"^\s*[-+]?\s*[\d\s.,]+(?:$)")

def clean_number(val):
    if pd.isna(val):
        return val
    s = str(val).strip()
    s = re.sub(r"[^\d\-,.\s]", "", s)
    s = s.replace(" ", "")
    if s.count(",") > 0 and s.count(".") == 0:
        s = s.replace(",", ".")
    elif s.count(".") > 1 and s.count(",") == 0:
        parts = s.split(".")
        s = "".join(parts[:-1]) + "." + parts[-1]
    try:
        return float(s) if "." in s else int(s)
    except Exception:
        return str(val).strip()

def clean_text(val):
    if not isinstance(val, str):
        return val
    text = val.strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s/\-_.%,:;()&]", "", text, flags=re.UNICODE)
    replacements = {
        "T otal": "Total",
        "T0tal": "Total",
        "Tơtal": "Total",
        "2O24": "2024",
        "lién": "liên",
        "hiém": "hiểm",
    }
    for wrong, right in replacements.items():
        text = text.replace(wrong, right)
    return text

def clean_dataframe(df):
    df_new = df.copy()
    df_new.columns = [clean_text(str(c)) for c in df_new.columns]
    extra_cols = []
    for col in df_new.columns:
        if col.lower() == "extra_values":
            split_extras = df_new[col].fillna("").astype(str).apply(
                lambda x: [p.strip() for p in x.split(";") if p.strip()]
            )
            max_len = split_extras.map(len).max() if not split_extras.empty else 0
            for i in range(max_len):
                new_col = f"extra{i+1}"
                def _get_clean(lst):
                    if i < len(lst):
                        val = lst[i]
                        if NUM_PATTERN.match(val):
                            return clean_number(val)
                        else:
                            return clean_text(val)
                    return ""
                df_new[new_col] = split_extras.apply(_get_clean)
                extra_cols.append(new_col)
            df_new.drop(columns=[col], inplace=True)
        else:
            df_new[col] = df_new[col].apply(
                lambda x: clean_number(x) if NUM_PATTERN.match(str(x).strip()) else clean_text(x)
            )
    if extra_cols:
        other_cols = [c for c in df_new.columns if c not in extra_cols]
        df_new = df_new[other_cols + extra_cols]
    return df_new

def process_all(choice="a"):
    if os.path.exists(CLEAN_DIR):
        if choice == "y":
            shutil.rmtree(CLEAN_DIR)
            print(f"🗑️ Đã xoá {CLEAN_DIR}")
        elif choice == "n":
            print("⏭️ Bỏ qua clean.")
            return
        elif choice == "a":
            print(f"➕ Giữ {CLEAN_DIR}, clean thêm file mới.")
        else:
            print("❌ Lựa chọn không hợp lệ, bỏ qua.")
            return
    ensure_dir(CLEAN_DIR)

    excel_files = glob.glob(os.path.join(RAW_DIR, "**", "*.xlsx"), recursive=True)
    text_files  = glob.glob(os.path.join(RAW_DIR, "**", "*.txt"),  recursive=True)
    json_files  = glob.glob(os.path.join(RAW_DIR, "**", "*.json"), recursive=True)

    for f in excel_files:
        rel_path = os.path.relpath(f, RAW_DIR)
        out_path = os.path.join(CLEAN_DIR, rel_path)
        ensure_dir(os.path.dirname(out_path))
        try:
            df = pd.read_excel(f, engine="openpyxl")
            df_clean = clean_dataframe(df)
            df_clean.to_excel(out_path, index=False)
            print(f"✅ Cleaned Excel: {out_path}")
        except Exception as e:
            print(f"⚠️ Lỗi xử lý {f}: {e}")

    for f in text_files:
        rel_path = os.path.relpath(f, RAW_DIR)
        out_path = os.path.join(CLEAN_DIR, rel_path)
        ensure_dir(os.path.dirname(out_path))
        try:
            with open(f, "r", encoding="utf-8") as fr:
                lines = fr.readlines()
            cleaned = [clean_text(line) for line in lines if line.strip()]
            with open(out_path, "w", encoding="utf-8") as fw:
                fw.write("\n".join(cleaned))
            print(f"📄 Cleaned Text: {out_path}")
        except Exception as e:
            print(f"⚠️ Lỗi xử lý text {f}: {e}")

    for f in json_files:
        rel_path = os.path.relpath(f, RAW_DIR)
        out_path = os.path.join(CLEAN_DIR, rel_path)
        ensure_dir(os.path.dirname(out_path))
        shutil.copy(f, out_path)
        print(f"📑 Copy Metadata JSON: {out_path}")

def process_file(file_path, output_path=None):
    """Clean riêng một file"""
    if file_path.endswith(".xlsx"):
        df = pd.read_excel(file_path, engine="openpyxl")
        df_clean = clean_dataframe(df)
        out_path = output_path or file_path.replace(RAW_DIR, CLEAN_DIR)
        ensure_dir(os.path.dirname(out_path))
        df_clean.to_excel(out_path, index=False)
        print(f"✅ Cleaned Excel: {out_path}")
    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as fr:
            lines = fr.readlines()
        cleaned = [clean_text(line) for line in lines if line.strip()]
        out_path = output_path or file_path.replace(RAW_DIR, CLEAN_DIR)
        ensure_dir(os.path.dirname(out_path))
        with open(out_path, "w", encoding="utf-8") as fw:
            fw.write("\n".join(cleaned))
        print(f"📄 Cleaned Text: {out_path}")
    elif file_path.endswith(".json"):
        out_path = output_path or file_path.replace(RAW_DIR, CLEAN_DIR)
        ensure_dir(os.path.dirname(out_path))
        shutil.copy(file_path, out_path)
        print(f"📑 Copy Metadata JSON: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean OCR raw outputs")
    parser.add_argument("--mode", type=str, default="all", choices=["all", "file"], help="Chạy clean toàn bộ folder hay chỉ 1 file")
    parser.add_argument("--file", type=str, help="Đường dẫn file khi mode=file")
    parser.add_argument("--choice", type=str, default="a", choices=["y","n","a"], help="Xử lý nếu output đã tồn tại (y = xóa, a = append, n = skip)")
    args = parser.parse_args()

    if args.mode == "all":
        process_all(choice=args.choice)
    elif args.mode == "file":
        if not args.file:
            print("❌ Cần truyền --file khi mode=file")
        else:
            process_file(args.file)
