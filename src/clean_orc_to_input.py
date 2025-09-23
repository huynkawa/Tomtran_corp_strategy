# 📁 src/clean_ocr.py
import os
import re
import pandas as pd
import yaml
from rapidfuzz import process

# ==== Cấu hình ==== 
SELECTED_DIR = "outputs/clean_orc_raw_output"
CLEAN_DIR = "inputs/cleaned_scan_input"
RULE_FILE = "configs/ocr_fix_rules.yaml"
LOG_FILE = "logs/ocr_unknown_words.txt"

os.makedirs(CLEAN_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)

# ==== Load rules từ YAML (chỉ dùng nếu advanced) ==== 
if os.path.exists(RULE_FILE):
    with open(RULE_FILE, "r", encoding="utf-8") as f:
        rules = yaml.safe_load(f)
    TEXT_CORRECTIONS = {r["wrong"].lower(): r["right"] for r in rules.get("text_corrections", [])}
    DOMAIN_DICT = list(TEXT_CORRECTIONS.values())
else:
    TEXT_CORRECTIONS = {}
    DOMAIN_DICT = []

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

# ==== Clean text (basic + advanced) ==== 
def clean_text(val, mode="advanced"):
    if not isinstance(val, str): return val
    text = val.strip()

    # Basic fix
    text = text.replace("T otal", "Total").replace("T0tal", "Total")
    text = text.replace("Tơtal", "Total").replace("2O24", "2024")
    text = re.sub(r"\s+", " ", text)

    if mode == "basic":
        return text

    # Advanced: dùng rule YAML + fuzzy
    lower_text = text.lower()
    for wrong, right in TEXT_CORRECTIONS.items():
        if wrong in lower_text:
            lower_text = lower_text.replace(wrong, right)

    words = lower_text.split()
    cleaned, unknowns = [], []
    for w in words:
        if DOMAIN_DICT:
            match, score, _ = process.extractOne(w, DOMAIN_DICT)
            if score > 85:
                cleaned.append(match)
            else:
                cleaned.append(w)
                unknowns.append(w)
        else:
            cleaned.append(w)

    if unknowns:
        with open(LOG_FILE, "a", encoding="utf-8") as logf:
            logf.write(" ".join(unknowns) + "\n")

    return " ".join(cleaned)

# ==== Clean DataFrame ==== 
def clean_dataframe(df, mode="advanced"):
    df_new = df.copy()
    for col in df_new.columns:
        df_new[col] = df_new[col].apply(
            lambda x: clean_number(x) if str(x).replace(".", "").isdigit() else clean_text(str(x), mode=mode)
        )
    return df_new

# ==== Xử lý tất cả file trong selected ==== 
def process_selected_files(mode="advanced"):
    for root, _, files in os.walk(SELECTED_DIR):
        for fname in files:
            fpath = os.path.join(root, fname)

            # Tính relative path từ selected → clean_scan
            rel_path = os.path.relpath(fpath, SELECTED_DIR)
            out_path = os.path.join(CLEAN_DIR, rel_path)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            if fname.endswith(".xlsx"):
                try:
                    df = pd.read_excel(fpath)
                    df_clean = clean_dataframe(df, mode=mode)
                    out_path = out_path.replace(".xlsx", "_clean.xlsx")
                    df_clean.to_excel(out_path, index=False)
                    print(f"✅ Cleaned Excel → {out_path}")
                except Exception as e:
                    print(f"⚠️ Lỗi xử lý Excel {fname}: {e}")

            elif fname.endswith(".txt"):
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                    cleaned = [clean_text(line, mode=mode) for line in lines if line.strip()]
                    out_path = out_path.replace(".txt", "_clean.csv")
                    pd.DataFrame({"text": cleaned}).to_csv(out_path, index=False)
                    print(f"✅ Cleaned Text → CSV: {out_path}")
                except Exception as e:
                    print(f"⚠️ Lỗi xử lý Text {fname}: {e}")

if __name__ == "__main__":
    # Có thể đổi "basic" nếu chỉ muốn clean nhẹ
    process_selected_files(mode="advanced")
