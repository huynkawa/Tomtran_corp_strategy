# p1c_text_validate_totals.py
import os, re, glob
from decimal import Decimal
from src.p1c_clean10_rulename import RULES, FORMULAS, NAME_RULES

INPUT_DIR = r"D:\1.TLAT\3. ChatBot_project\1_Insurance_Strategy - Copy\outputs\p1b_clean10_ocr_text_test1"
OUTPUT_DIR = r"D:\1.TLAT\3. ChatBot_project\1_Insurance_Strategy - Copy\outputs\p1c_text_validate_totals"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================
# Utils
# =============================

def fix_num_errors(val: str) -> str:
    """Sửa số OCR lỗi: loại ký tự lạ, chuẩn nghìn"""
    if not val:
        return ""
    val = re.sub(r"[^\d]", "", val)  # bỏ ký tự không phải số
    if len(val) > 12:  # dính liền, thêm dấu chấm nghìn
        parts = []
        while len(val) > 3:
            parts.insert(0, val[-3:])
            val = val[:-3]
        if val:
            parts.insert(0, val)
        val = ".".join(parts)
    return val


def clean_num(val: str) -> Decimal:
    """Chuẩn hóa chuỗi số -> Decimal (an toàn hơn)"""
    if not val:
        return Decimal(0)
    val = fix_num_errors(val)
    try:
        return Decimal(val)
    except Exception:
        # Nếu OCR sinh chuỗi không hợp lệ, coi như 0
        return Decimal(0)


def parse_txt(path: str):
    """
    Đọc file txt -> dict
    Trả về:
        codes: {mã số -> (số cuối năm, số đầu năm)}
        names: {tên chỉ tiêu -> (số cuối năm, số đầu năm)}
    """
    codes, names = {}, {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # match theo mã số
            m = re.match(r"^(\d+[a-zA-Z0-9\.]*)\s+(.+?)\s+([\d\.\,]+)\s+([\d\.\,]+)$", line)
            if m:
                code, name, end, begin = m.groups()
                codes[code] = (clean_num(end), clean_num(begin))
                names[name.lower()] = (clean_num(end), clean_num(begin))
                continue

            # fallback: chỉ có tên + 2 số
            m2 = re.match(r"^(.+?)\s+([\d\.\,]+)\s+([\d\.\,]+)$", line)
            if m2:
                name, end, begin = m2.groups()
                names[name.lower()] = (clean_num(end), clean_num(begin))

    return codes, names

# =============================
# Formula checker
# =============================

def check_formula(expr: str, codes: dict, filename: str, code: str):
    """
    Kiểm tra công thức đặc biệt (VD: 50 = 10 + 12 - 20 ...)
    expr: chuỗi công thức trong FORMULAS
    """
    # tách vế trái và vế phải
    left, right = expr.split("=")
    left = left.strip()
    right_expr = right.strip()

    # tính toán cho số cuối năm & số đầu năm
    def eval_side(expr_str, idx):
        # thay mỗi số bằng giá trị thực tế từ codes
        tokens = re.findall(r"[\+\-]|\d+[a-zA-Z0-9\.]*", expr_str)
        total = Decimal(0)
        sign = 1
        for tok in tokens:
            if tok == "+":
                sign = 1
            elif tok == "-":
                sign = -1
            else:
                if tok in codes:
                    total += sign * codes[tok][idx]
                else:
                    # nếu thiếu code thì coi = 0
                    total += 0
        return total

    if left not in codes:
        return f"[{filename}] ⚠ Thiếu mã {left} trong dữ liệu để kiểm tra {expr}"

    end, begin = codes[left]
    sum_end = eval_side(right_expr, 0)
    sum_begin = eval_side(right_expr, 1)

    if end != sum_end or begin != sum_begin:
        return f"[{filename}] LỆCH công thức {expr}: {end}/{begin} ≠ {sum_end}/{sum_begin}"
    return None

# =============================
# Validate
# =============================

def validate(codes: dict, names: dict, filename: str):
    logs = []

    # --- RULES theo mã số ---
    for rule in RULES:
        if len(rule) == 2:
            parent, children = rule
            if parent not in codes:
                continue
            if not all(c in codes for c in children):
                continue

            sum_end = sum(codes[c][0] for c in children)
            sum_begin = sum(codes[c][1] for c in children)
            end, begin = codes[parent]

            if end != sum_end or begin != sum_begin:
                logs.append(f"[{filename}] LỆCH mã {parent}: {end}/{begin} ≠ {sum_end}/{sum_begin}")

        else:
            parent, children, formula_key = rule
            if formula_key in FORMULAS:
                err = check_formula(FORMULAS[formula_key], codes, filename, parent)
                if err:
                    logs.append(err)

    # --- NAME_RULES fallback ---
    for parent, child_names in NAME_RULES.items():
        if parent not in names:
            continue
        if not all(c in names for c in child_names):
            continue

        sum_end = sum(names[c][0] for c in child_names if c in names)
        sum_begin = sum(names[c][1] for c in child_names if c in names)
        end, begin = names[parent]

        if end != sum_end or begin != sum_begin:
            logs.append(f"[{filename}] LỆCH tên '{parent}': {end}/{begin} ≠ {sum_end}/{sum_begin}")

    return logs

# =============================
# Main
# =============================

def main():
    for path in glob.glob(os.path.join(INPUT_DIR, "*.txt")):
        fname = os.path.basename(path)
        codes, names = parse_txt(path)

        if not codes and not names:
            print(f"⚠ {fname}: KHÔNG parse được dòng nào (trang rỗng hoặc OCR lỗi)")
            continue

        logs = validate(codes, names, fname)

        # ghi file validate
        out_validate = os.path.join(OUTPUT_DIR, fname.replace(".txt", "_validate.txt"))
        with open(out_validate, "w", encoding="utf-8") as f:
            if logs:
                f.write("\n".join(logs))
            else:
                f.write("✅ Tất cả công thức khớp")

        # ghi file clean
        out_clean = os.path.join(OUTPUT_DIR, fname.replace(".txt", "_clean.txt"))
        with open(out_clean, "w", encoding="utf-8") as f:
            for code, (end, begin) in codes.items():
                f.write(f"{code}\t{end}\t{begin}\n")
            for name, (end, begin) in names.items():
                if name not in codes:
                    f.write(f"{name}\t{end}\t{begin}\n")

        print(f"✅ Xong {fname}, log = {len(logs)} lỗi")

if __name__ == "__main__":
    main()
