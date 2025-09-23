#src/preprocess_inputs.py tự động:

# Chạy OCR nếu là file scan.

# Copy file sạch từ raw_clean_docs/ → orc_cleaned_data/.

# 
#  📁 src/clean_ocr_advanced.py
import os
import re
import pandas as pd
import yaml
from rapidfuzz import process

# ==== Cấu hình đường dẫn ====
SELECTED_DIR = "outputs/orc_raw_output_selected"
FINAL_DIR = "inputs/orc_cleaned_data"
RULE_FILE = "configs/ocr_fix_rules.yaml"
LOG_FILE = "logs/ocr_unknown_words.txt"

os.makedirs(FINAL_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)

# ==== Load rules từ YAML ====
with open(RULE_FILE, "r", encoding="utf-8") as f:
    rules = yaml.safe_load(f)
TEXT_CORRECTIONS = {r["wrong"].lower(): r["right"] for r in rules.get("text_corrections", [])}
DOMAIN_DICT = list(TEXT_CORRECTIONS.values())

# ==== Clean number ====
def clean_number(val):
    if pd.isna(val): return val
    s = str(val)
    s = re.sub(r"[^\d\-,.]", "", s)
    s = s.replace(",", ".")
    if s.count(".") > 1:
        parts = s.split(".")
        s = "".join(parts[:-1]) + "." + parts[-1]
    try:
        if "." not in s: return int(s)
        return float(s)
    except:
        return val

# ==== Clean text with fuzzy + rules ====
def clean_text(val):
    if not isinstance(val, str): return val
    val = val.strip().lower()
    for wrong, right in TEXT_CORRECTIONS.items():
        if wrong in val:
            val = val.replace(wrong, right)

    words = val.split()
    cleaned = []
    unknowns = []
    for w in words:
        match, score, _ = process.extractOne(w, DOMAIN_DICT)
        if score > 85:
            cleaned.append(match)
        else:
            cleaned.append(w)
            unknowns.append(w)
    if unknowns:
        with open(LOG_FILE, "a", encoding="utf-8") as logf:
            logf.write(" ".join(unknowns) + "\n")
    return " ".join(cleaned)

# ==== Clean DataFrame ====
def clean_dataframe(df):
    df_new = df.copy()
    for col in df_new.columns:
        df_new[col] = df_new[col].apply(
            lambda x: clean_number(x) if str(x).replace(".", "").isdigit() else clean_text(str(x))
        )
    return df_new

# ==== Clean tất cả các file được OCR ====
def process_selected_files():
    for fname in os.listdir(SELECTED_DIR):
        fpath = os.path.join(SELECTED_DIR, fname)

        if fname.endswith(".xlsx"):
            df = pd.read_excel(fpath)
            df_clean = clean_dataframe(df)
            out_path = os.path.join(FINAL_DIR, fname.replace(".xlsx", "_clean.xlsx"))
            df_clean.to_excel(out_path, index=False)
            print(f"✅ Cleaned Excel: {out_path}")

        elif fname.endswith(".txt"):
            with open(fpath, "r", encoding="utf-8") as f:
                lines = f.readlines()
            cleaned = [clean_text(line) for line in lines if line.strip()]
            out_path = os.path.join(FINAL_DIR, fname.replace(".txt", "_clean.csv"))
            pd.DataFrame({"text": cleaned}).to_csv(out_path, index=False)
            print(f"✅ Cleaned Text → CSV: {out_path}")

if __name__ == "__main__":
    process_selected_files()
