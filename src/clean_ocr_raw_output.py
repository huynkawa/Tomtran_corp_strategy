# 📁 src/clean_raw_output.py
import os
import re
import glob
import shutil
import pandas as pd

RAW_DIR = r"outputs/orc_raw_output"           # dữ liệu OCR thô
CLEAN_DIR = r"outputs/clean_orc_raw_output"   # dữ liệu sau khi clean bước 1

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

# === Clean số cơ bản ===
def clean_number(val):
    if pd.isna(val):
        return val
    s = str(val).strip()

    # Chỉ giữ số, dấu . , -
    s = re.sub(r"[^\d,.\-]", "", s)

    try:
        # Trường hợp có cả , và . → giả định , là nghìn, . là thập phân
        if "," in s and "." in s:
            s = s.replace(",", "")
            num = float(s)

        # Trường hợp chỉ có , → coi là nghìn
        elif "," in s and "." not in s:
            s = s.replace(",", "")
            num = int(s)
        
        # Trường hợp chỉ có . → phân biệt nghìn vs thập phân
        elif "." in s and "," not in s:
            parts = s.split(".")
            # nếu các phần sau dấu . toàn có 3 chữ số → dấu . là nghìn
            if all(len(p) == 3 for p in parts[1:]):
                s = "".join(parts)
                num = int(s)
            else:
                num = float(s)
        else:
            num = int(s)

        # Format lại theo chuẩn: nghìn = ',', thập phân = '.'
        if isinstance(num, int):
            return f"{num:,}".replace(",", ".")  # 1.000.000
        else:
            int_part, dec_part = str(num).split(".")
            int_part_fmt = f"{int(int_part):,}".replace(",", ".")
            return f"{int_part_fmt}.{dec_part}"

    except:
        return val


# === Clean text cơ bản ===
def clean_text(val):
    if not isinstance(val, str):
        return val
    text = val.strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s\-/.,]", "", text)

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

# === Clean DataFrame Excel ===
def clean_dataframe(df):
    df_new = df.copy()
    df_new.columns = [clean_text(str(c)) for c in df_new.columns]
    for col in df_new.columns:
        df_new[col] = df_new[col].apply(
            lambda x: clean_number(x) if str(x).replace(".", "").replace(",", "").isdigit() else clean_text(x)
        )
    return df_new

# === Clean toàn bộ file trong RAW_DIR ===
def process_all():
    if os.path.exists(CLEAN_DIR):
        shutil.rmtree(CLEAN_DIR)
    ensure_dir(CLEAN_DIR)

    excel_files = glob.glob(os.path.join(RAW_DIR, "**", "*.xlsx"), recursive=True)
    text_files = glob.glob(os.path.join(RAW_DIR, "**", "*.txt"), recursive=True)

    for f in excel_files:
        rel_path = os.path.relpath(f, RAW_DIR)
        out_path = os.path.join(CLEAN_DIR, rel_path)
        ensure_dir(os.path.dirname(out_path))
        try:
            df = pd.read_excel(f)
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

if __name__ == "__main__":
    process_all()
