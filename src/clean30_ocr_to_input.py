# 📁 src/clean30_ocr_to_input.py — V3 (merge multi-line, parent-child hierarchy)
import os, re, shutil, yaml
import pandas as pd
from rapidfuzz import process
from collections import defaultdict

# ==== Cấu hình ====
SELECTED_DIR = "outputs/clean_orc_raw_output"
CLEAN_DIR    = "inputs/cleaned_scan_input"
RULE_FILE    = "configs/ocr_fix_rules.yaml"
LOG_FILE     = "logs/ocr_unknown_words.txt"

os.makedirs(CLEAN_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)

# ==== Load rules từ YAML (advanced) ====
if os.path.exists(RULE_FILE):
    with open(RULE_FILE, "r", encoding="utf-8") as f:
        rules = yaml.safe_load(f) or {}
    TEXT_CORRECTIONS = {str(r.get("wrong","")).lower(): r.get("right","") for r in rules.get("text_corrections", [])}
    DOMAIN_DICT = [str(v).lower() for v in TEXT_CORRECTIONS.values() if v]
else:
    TEXT_CORRECTIONS = {}
    DOMAIN_DICT = []

# ==== Nhận diện số ====
NUM_PATTERN = re.compile(r"^\s*[-+]?\s*[\d\s.,]+$")

def clean_number(val):
    if pd.isna(val): return val
    s = str(val).strip()
    s = re.sub(r"[^\d\-,.\s]", "", s).replace(" ", "")
    if s.count(",")>0 and s.count(".")==0: s = s.replace(",", ".")
    elif s.count(".")>1 and s.count(",")==0:
        parts = s.split("."); s = "".join(parts[:-1]) + "." + parts[-1]
    try:
        return float(s) if "." in s else int(s)
    except: return str(val).strip()

# ==== Clean text (basic + advanced) ====
def clean_text(val, mode="advanced"):
    if not isinstance(val, str): return val
    text = re.sub(r"\s+", " ", val.strip())
    text = (text.replace("T otal","Total").replace("T0tal","Total")
                 .replace("Tơtal","Total").replace("2O24","2024"))
    if mode == "basic": return text

    lower_text = text.lower()
    for wrong, right in TEXT_CORRECTIONS.items():
        if wrong and wrong in lower_text:
            lower_text = lower_text.replace(wrong, str(right))

    words = lower_text.split()
    cleaned, unknowns = [], []
    for w in words:
        if DOMAIN_DICT and re.search(r"[a-zA-ZÀ-ỹ]", w):
            match = process.extractOne(w, DOMAIN_DICT)
            if match and match[1] > 85: cleaned.append(match[0])
            else: cleaned.append(w); unknowns.append(w)
        else:
            cleaned.append(w)
    if unknowns:
        with open(LOG_FILE, "a", encoding="utf-8") as logf:
            logf.write(" ".join(unknowns) + "\n")
    return " ".join(cleaned)

# ==== Chuẩn hóa kỳ end/start từ header/ngày ====
END_PAT   = re.compile(r"(số\s*cuối\s*năm|cuối\s*năm|ending|end\s*of\s*year|as\s+at)", re.I)
START_PAT = re.compile(r"(số\s*đầu\s*năm|đầu\s*năm|begin|start\s*of\s*year|opening)", re.I)
DATE_PAT  = re.compile(r"(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})")

def _parse_date(s: str):
    from datetime import datetime
    for fmt in ("%d/%m/%Y","%d-%m-%Y","%d/%m/%y","%d-%m-%y"):
        try:
            return datetime.strptime(s, fmt)
        except: pass
    return None

def normalize_period_by_dates(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    date_cols = []
    for c in cols:
        m = DATE_PAT.search(str(c))
        if m:
            dt = _parse_date(m.group(1))
            if dt: date_cols.append((c, dt))
    if len(date_cols) >= 2:
        date_cols.sort(key=lambda x: x[1])  # tăng dần
        start_col = date_cols[0][0]
        end_col   = date_cols[-1][0]
        df = df.rename(columns={start_col: "start", end_col: "end"})
    return df

def normalize_period_columns(df):
    df2 = df.copy()
    df2 = normalize_period_by_dates(df2)
    rename = {}
    for c in df2.columns:
        s = str(c).lower()
        if END_PAT.search(s):   rename[c] = "end"
        elif START_PAT.search(s): rename[c] = "start"
    if rename: df2 = df2.rename(columns=rename)
    return df2

def heuristic_three_cols(df):
    df2 = df.copy()
    if df2.shape[1] == 3:
        c0,c1,c2 = df2.columns[:3]
        def num_rate(s): return float(pd.Series([bool(NUM_PATTERN.match(str(x).strip())) for x in s]).mean())
        if num_rate(df2[c1])>0.6 and num_rate(df2[c2])>0.6:
            df2 = df2.rename(columns={c0:"line_item", c1:"end", c2:"start"})
    return df2

def ensure_unit_columns(df, currency_default="VND", scale_default="đồng"):
    df2 = df.copy()
    if "currency" not in [str(c) for c in df2.columns]: df2["currency"] = currency_default
    if "scale" not in [str(c) for c in df2.columns]: df2["scale"] = scale_default
    return df2

# ==== Chuẩn hoá header & phân cấp dòng ====
ROMAN_PAT = re.compile(r"^(i|ii|iii|iv|v|vi|vii|viii|ix|x)\.?\s", re.I)

def infer_level_and_flags(code: str, line_item: str):
    c = (code or "").strip()
    li = (line_item or "").strip()

    # nhóm theo roman hoặc đầu mục chữ
    if ROMAN_PAT.match(li) or re.match(r"^[A-Z]\.\s", li):
        return "group", True, False

    if re.fullmatch(r"\d{3}", c):
        if c.endswith("00"):
            return "group", True, False
        else:
            return "item", False, False
    if re.fullmatch(r"\d+(\.\d+)*", c):
        return "item", False, False
    return "item", False, False

def merge_same_code_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Ghép nhiều dòng cùng code thành một dòng"""
    if "code" not in df.columns: return df
    merged = []
    for code, group in df.groupby("code", dropna=False):
        if len(group) == 1:
            merged.append(group.iloc[0])
        else:
            base = group.iloc[0].copy()
            # nối tên chỉ tiêu
            base["line_item"] = " ".join(str(x) for x in group["line_item"] if pd.notna(x))
            # ưu tiên giá trị số ở dòng nào có
            for col in ("end","start"):
                vals = [v for v in group[col] if str(v).strip()!=""] if col in group.columns else []
                base[col] = vals[0] if vals else ""
            merged.append(base)
    return pd.DataFrame(merged).reset_index(drop=True)

def assign_parent_child(df: pd.DataFrame) -> pd.DataFrame:
    """Xác định parent_code cho mỗi code"""
    df2 = df.copy()
    parents = []
    code_list = df2["code"].astype(str).tolist()

    for c in code_list:
        parent = ""
        if re.fullmatch(r"\d+\.\d+", c):  # ví dụ 1.1 → 1
            parent = c.rsplit(".",1)[0]
        elif re.fullmatch(r"\d{3}\.\d+", c):  # 131.2 → 131
            parent = c.rsplit(".",1)[0]
        elif re.fullmatch(r"\d+", c):  # mục số → có thể thuộc nhóm roman ở trên
            parent = ""
        parents.append(parent)
    df2["parent_code"] = parents

    # build children list
    children_map = defaultdict(list)
    for idx,row in df2.iterrows():
        p = row.get("parent_code","")
        if p:
            children_map[p].append(row["code"])
    df2["children_codes"] = df2["code"].apply(lambda x: children_map.get(x, []))
    return df2


# hàm validate_subtotals để kiểm tra các quan hệ end = sum(children) và start = sum(children)
def validate_subtotals(df: pd.DataFrame, tolerance_ratio=0.005, tolerance_abs=1.0, log_file="logs/subtotal_mismatch.txt") -> pd.DataFrame:
    """
    Kiểm tra subtotal: so sánh giá trị end/start của cha với tổng các con.
    - tolerance_ratio: cho phép sai số % (mặc định 0.5%)
    - tolerance_abs: cho phép sai số tuyệt đối (mặc định ±1)
    - log_file: đường dẫn file log ghi lại các mismatch
    """
    df2 = df.copy()
    code_map = {str(r["code"]): r for _, r in df2.iterrows()}

    end_checks, start_checks = [], []

    # mở log file để ghi
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, "w", encoding="utf-8") as logf:
        for idx, row in df2.iterrows():
            children = row.get("children_codes", [])
            if not children:
                end_checks.append("")
                start_checks.append("")
                continue

            for label, checks in [("end", end_checks), ("start", start_checks)]:
                parent_val = row.get(f"{label}_num", None)
                if parent_val is None or parent_val == "":
                    checks.append("")
                    continue

                child_vals = [code_map[c].get(f"{label}_num") for c in children if c in code_map]
                child_vals = [v for v in child_vals if isinstance(v, (int, float))]
                if not child_vals:
                    checks.append("")
                    continue

                total = sum(child_vals)
                diff = abs(total - parent_val)
                ok = (diff <= tolerance_abs) or (abs(diff / parent_val) <= tolerance_ratio if parent_val != 0 else False)
                mark = "✅" if ok else f"❌ ({parent_val} vs {total})"
                checks.append(mark)

                # nếu mismatch thì ghi log
                if not ok:
                    logf.write(f"[{label.upper()}] Code {row.get('code','')} ({row.get('line_item','')}) "
                               f"= {parent_val}, sum(children {children}) = {total}\n")

    df2["end_check"] = end_checks
    df2["start_check"] = start_checks
    return df2




def mark_hierarchy(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    if "line_item" not in df2.columns:
        lens = {c: df2[c].astype(str).str.len().mean() for c in df2.columns}
        df2 = df2.rename(columns={max(lens, key=lens.get): "line_item"})
    if "code" not in df2.columns: df2["code"] = ""

    df2["line_item"] = df2["line_item"].apply(lambda s: clean_text(s, mode="advanced"))
    df2["line_item_norm"] = df2["line_item"].apply(lambda s: re.sub(r"\s+"," ",s.lower().strip()))

    levels, totals, subtotals = [], [], []
    for _, r in df2.iterrows():
        level, is_total, is_sub = infer_level_and_flags(str(r.get("code","")), str(r.get("line_item","")))
        levels.append(level); totals.append(is_total); subtotals.append(is_sub)
    df2["level"] = levels; df2["is_total"] = totals; df2["is_subtotal"] = subtotals

    # đánh dấu subtotal có con
    codes = df2["code"].astype(str).tolist()
    for idx,c in enumerate(codes):
        if c and any(cc.startswith(c+".") for cc in codes if cc!=c):
            df2.loc[idx,"is_subtotal"] = True

    # tiện cho validate
    for col in ("end","start"):
        if col in df2.columns:
            df2[f"{col}_num"] = df2[col].apply(clean_number)

    # merge + parent-child

    df2 = merge_same_code_rows(df2)
    df2 = assign_parent_child(df2)
    df2 = validate_subtotals(df2)   # có log file
    return df2



# ==== Clean DataFrame tổng hợp ====
def clean_dataframe(df, mode="advanced"):
    df_new = df.copy()
    for col in df_new.columns:
        df_new[str(col) + "_raw"] = df_new[col]

    for col in df_new.columns:
        if str(col).endswith("_raw"): continue
        def _clean(x):
            xs = "" if x is None else str(x)
            if NUM_PATTERN.match(xs.strip()): return clean_number(xs)
            return clean_text(xs, mode=mode)
        df_new[col] = df_new[col].apply(_clean)

    df_new = normalize_period_columns(df_new)
    if "end" not in df_new.columns and "start" not in df_new.columns:
        df_new = heuristic_three_cols(df_new)

    df_new = ensure_unit_columns(df_new)
    df_new = mark_hierarchy(df_new)
    return df_new

# ==== Xử lý tất cả file ====
def process_selected_files(mode="advanced"):
    if os.path.exists(CLEAN_DIR):
        choice = input(f"⚠️ Folder {CLEAN_DIR} đã tồn tại. "
                       "Chọn y = xoá build lại, a = append thêm file mới, n = bỏ qua: ").strip().lower()
        if choice == "y":
            shutil.rmtree(CLEAN_DIR)
            print(f"🗑️ Đã xoá {CLEAN_DIR}")
        elif choice == "n":
            print("⏭️ Bỏ qua clean30.")
            return
        elif choice == "a":
            print(f"➕ Giữ {CLEAN_DIR}, clean thêm file mới.")
        else:
            print("❌ Lựa chọn không hợp lệ, bỏ qua.")
            return
    os.makedirs(CLEAN_DIR, exist_ok=True)

    for root, _, files in os.walk(SELECTED_DIR):
        for fname in files:
            fpath = os.path.join(root, fname)
            rel_path = os.path.relpath(fpath, SELECTED_DIR)
            out_path = os.path.join(CLEAN_DIR, rel_path)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            if fname.lower().endswith(".xlsx"):
                try:
                    df = pd.read_excel(fpath, engine="openpyxl")
                    df_clean = clean_dataframe(df, mode=mode)
                    out_xlsx = out_path.replace(".xlsx", "_clean.xlsx")
                    out_csv  = out_path.replace(".xlsx", "_clean.csv")
                    df_clean.to_excel(out_xlsx, index=False)
                    df_clean.to_csv(out_csv, index=False, encoding="utf-8-sig")
                    print(f"✅ Cleaned Excel → {out_xlsx}")
                    print(f"✅ Cleaned CSV   → {out_csv}")
                except Exception as e:
                    print(f"⚠️ Lỗi xử lý Excel {fname}: {e}")

            elif fname.lower().endswith(".txt"):
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        lines = [ln for ln in f.readlines() if ln.strip()]
                    cleaned = [clean_text(ln, mode=mode) for ln in lines]
                    out_txt = out_path.replace(".txt", "_clean.txt")
                    with open(out_txt, "w", encoding="utf-8") as fw: 
                        fw.write("\n".join(cleaned))
                    out_csv = out_path.replace(".txt", "_clean.csv")
                    pd.DataFrame({"text": cleaned}).to_csv(out_csv, index=False, encoding="utf-8-sig")
                    print(f"✅ Cleaned Text → {out_txt}")
                    print(f"✅ Cleaned Text → CSV: {out_csv}")
                except Exception as e:
                    print(f"⚠️ Lỗi xử lý Text {fname}: {e}")

            elif fname.lower().endswith(".json"):
                try:
                    shutil.copy(fpath, out_path)
                    print(f"📑 Copy Metadata JSON → {out_path}")
                except Exception as e:
                    print(f"⚠️ Lỗi copy Metadata {fname}: {e}")
if __name__ == "__main__":
    process_selected_files(mode="advanced")
