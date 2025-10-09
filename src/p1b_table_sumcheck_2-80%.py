# -*- coding: utf-8 -*-
"""
TXT → Table (UNIVERSAL, derived from your base)

Giữ nguyên tinh thần của file gốc, nhưng mở rộng để dùng chung cho nhiều dạng báo cáo OCR:
- Tự động/tuỳ chọn số cột giá trị ở cuối dòng (1..4), mặc định auto=2 (kiểu BCTC 2 kỳ).
- Bắt số tiền mạnh (không gộp hai số), hỗ trợ (123.456) ⇒ -123456, -1.234.567, 90000000000…
- Nhận tuỳ chọn % và/hoặc số nhỏ (<100k) nếu cần.
- “Mã số” nhận chắc tay (tối đa 2 dấu chấm, tổng chữ số ≤6, loại mã trông như số tiền).
- Fallback khi file OCR dính 1 dòng dài (tách theo “Mã số”).
- Mirror folder con; prompt trùng file: y/n/a.

Cách chạy (giống file gốc):
    python -m src.p1b_table_sumcheck_2
Hoặc chạy trực tiếp file này:
    python p1b_table_sumcheck_2_UNI_FROM_BASE.py --in "IN_DIR" --out "OUT_DIR"

Tham số thêm:
    --value-cols N        Cố định N cột giá trị ở cuối dòng (1..4). Nếu bỏ qua → AUTO.
    --labels CSV          Nhãn cho các cột giá trị, ví dụ: "Số cuối năm,Số đầu năm"
    --allow-percent       Nhận số dạng phần trăm làm giá trị (ví dụ 12,5% → 12.5%)
    --accept-small        Nhận số nhỏ (<100k) (mặc định bỏ qua để giảm nhiễu OCR)
    --fallback-only       Chỉ dùng bộ tách theo segment toàn văn bản (OCR 1 dòng dài)
"""

from __future__ import annotations
import os
import re
import sys
import argparse
import statistics
from typing import Dict, List, Optional, Tuple

# ====== MẶC ĐỊNH NHƯ FILE GỐC ======
DEFAULT_IN_DIR  = r"D:\1.TLAT\3. ChatBot_project\1_Insurance_Strategy - Copy\outputs\p1a_clean10_ocr_bctc"
DEFAULT_OUT_DIR = r"D:\1.TLAT\3. ChatBot_project\1_Insurance_Strategy - Copy\outputs\p1b_table_sumcheck_2"
DEFAULT_FMT = "txt"
DEFAULT_ON_EXIST = "ask"   # ask | overwrite | skip | append

# ====== Cấu hình & helpers ======
ELLIPSIS = "..."
MAX_CHI_TIEU_LENGTH = 120  # nới hơn file gốc

def truncate(text: str, max_len: int) -> str:
    text = text.strip()
    if len(text) <= max_len:
        return text
    return (text[: max_len - len(ELLIPSIS)].rstrip() + ELLIPSIS)

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

# Token số KHÔNG chứa khoảng trắng → không gộp 2 số
NUM_TOKEN = re.compile(r"\(?-?\d[\d\.,]*\)?")
PCT_TOKEN = re.compile(r"\d[\d\.,]*\s*%")

def normalize_money_token(tok: str, *, accept_small: bool=False) -> Optional[str]:
    if not tok:
        return None
    t = tok.strip()

    # (123.456) -> -123456
    if t.startswith("(") and t.endswith(")"):
        inner = t[1:-1]
        core = re.sub(r"[^\d]", "", inner)
        if core.isdigit():
            if core == "0" or accept_small or len(core) >= 6:
                return "-" + core
        return None

    # -1.234.567 | 1,234,567 | 9000000
    core = re.sub(r"[^\d\-]", "", t)
    if core in ("", "-"):
        return None
    signless = core.lstrip("-")
    if not signless.isdigit():
        return None
    if signless == "0":
        return core
    if accept_small or len(signless) >= 6:
        return core
    return None

def normalize_percent_token(tok: str) -> Optional[str]:
    if not tok:
        return None
    m = PCT_TOKEN.search(tok)
    if not m:
        return None
    s = m.group(0)
    s = s.replace(",", ".").replace(" ", "")
    s2 = s.rstrip("%")
    # bỏ chấm ngăn nghìn nội bộ nếu có
    s2 = re.sub(r"(?<=\d)\.(?=\d{3}\b)", "", s2)
    try:
        float(s2)
    except:
        return None
    return s2 + "%"

def clean_token(tok: Optional[str]) -> str:
    return tok or ""

# Một số từ khoá "header" để loại bỏ (giữ từ file gốc)
HEADER_GARBAGE = re.compile(
    r"(Công\s*ty|BẢNG\s*CÂN|BANG\s*CAN|ngày\s+\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{4}|Đơn\s* vị\s* tính|Don\s* vi\s* tinh|Thuyết|Thuyết minh|fninh)",
    flags=re.I
)

# ===== Mã số =====
CODE_HEAD = re.compile(r"^\s*(\d{1,3}(?:\.\d{1,3}){0,2}|[IVXLCDM]+|[A-Z]\.)\s+(.+)$", flags=re.I)
CODE_ANCHOR = re.compile(r"(?:^|\s)(\d{1,3}(?:\.\d{1,3}){0,2}|[IVXLCDM]+|[A-Z]\.)\s+", flags=re.I)

def looks_like_money_code(c: str) -> bool:
    parts = c.split(".")
    if len(parts) > 3:  # >2 dấu chấm
        return True
    digits = "".join(parts)
    if len(digits) > 6:
        return True
    if sum(1 for p in parts if len(p) == 3) >= 2 and len(parts) >= 3:
        return True
    return False

def split_code_and_name(text_part: str) -> Tuple[str, str]:
    s = text_part.strip()
    s = re.sub(r"^(Mã\s*số|Ma\s* so|Chi\s*tieu|Chỉ\s* tiêu)\s*[:\-]?\s*", "", s, flags=re.I)
    m = CODE_HEAD.match(s)
    if m:
        code = m.group(1).rstrip(".")
        name = m.group(2).strip()
        if looks_like_money_code(code):
            return "", s
        return code, name
    return "", s

# ===== Tách giá trị ở cuối dòng =====
def extract_tail_values(line: str, k: int, *, allow_percent: bool, accept_small: bool) -> Tuple[str, List[str]]:
    money_toks = NUM_TOKEN.findall(line)
    money_vals: List[str] = []
    for t in money_toks:
        n = normalize_money_token(t, accept_small=accept_small)
        if n is not None:
            money_vals.append(n)

    pct_vals: List[str] = []
    if allow_percent and len(money_vals) < k:
        for t in PCT_TOKEN.findall(line):
            p = normalize_percent_token(t)
            if p:
                pct_vals.append(p)

    vals = money_vals + pct_vals
    if len(vals) >= k:
        picked = vals[-k:]
        # cắt head trước token thô cuối cùng xuất hiện trong line (ưu tiên token số)
        last_tok = None
        if money_toks:
            last_tok = money_toks[-1]
        elif allow_percent:
            ptoks = PCT_TOKEN.findall(line)
            if ptoks:
                last_tok = ptoks[-1]
        if last_tok:
            idx = line.rfind(last_tok)
            head = line[:idx].rstrip()
        else:
            head = line
        return head, picked
    else:
        return line, []

# ===== Parser theo dòng (mở rộng từ file gốc) =====
def parse_line(line: str, k: int, *, allow_percent: bool, accept_small: bool) -> Optional[Dict[str, str]]:
    line = line.strip()
    if not line:
        return None
    if HEADER_GARBAGE.search(line):
        return None
    line = re.sub(r"\s+", " ", line)

    head, vals = extract_tail_values(line, k, allow_percent=allow_percent, accept_small=accept_small)
    code, name = split_code_and_name(head)
    name = re.sub(r"^[\d\.\s]{3,}\s*", "", name).strip()
    name = truncate(name, MAX_CHI_TIEU_LENGTH)

    if not (name or code or vals):
        return None

    row = {"Mã số": code, "Tên chỉ tiêu": name}
    for i, v in enumerate(vals[::-1], start=1):  # đảo để cột 1 là số gần cuối dòng
        row[f"Giá trị {i}"] = clean_token(v)
    return row

# ===== Fallback: tách theo segment toàn văn bản (OCR 1 dòng dài) =====
def find_segments_by_code(text: str) -> List[Tuple[int,int]]:
    spans = []
    starts = [m.start(1) for m in CODE_ANCHOR.finditer(text)]
    if not starts:
        return [(0, len(text))]
    for i, st in enumerate(starts):
        en = starts[i+1] if i+1 < len(starts) else len(text)
        spans.append((st, en))
    return spans

def parse_segment(seg: str, k: int, *, allow_percent: bool, accept_small: bool) -> Optional[Dict[str,str]]:
    head, vals = extract_tail_values(seg, k, allow_percent=allow_percent, accept_small=accept_small)
    if not vals:
        return None
    code, name = split_code_and_name(head)
    name = re.sub(r"^[\d\.\s]{3,}\s*", "", name).strip()
    name = truncate(name, MAX_CHI_TIEU_LENGTH)
    row = {"Mã số": code, "Tên chỉ tiêu": name}
    for i, v in enumerate(vals[::-1], start=1):
        row[f"Giá trị {i}"] = clean_token(v)
    return row

def parse_whole_text(text: str, k: int, *, allow_percent: bool, accept_small: bool) -> List[Dict[str,str]]:
    rows: List[Dict[str,str]] = []
    for st, en in find_segments_by_code(text):
        seg = text[st:en]
        rec = parse_segment(seg, k, allow_percent=allow_percent, accept_small=accept_small)
        if rec and (rec.get("Tên chỉ tiêu") or any(rec.get(f"Giá trị {i}", "") for i in range(1, k+1))):
            rows.append(rec)
    if not rows:
        rec = parse_segment(text, k, allow_percent=allow_percent, accept_small=accept_small)
        if rec:
            rows.append(rec)
    return rows

# ===== Auto detect số cột giá trị =====
def detect_value_cols(lines: List[str], *, allow_percent: bool, accept_small: bool) -> int:
    samples = lines[:200] if len(lines) > 200 else lines
    counts = []
    for ln in samples:
        _, vals = extract_tail_values(ln, 4, allow_percent=allow_percent, accept_small=accept_small)
        counts.append(len(vals))
    try:
        mode = statistics.mode([c for c in counts if c in (1,2,3,4)] or [2])
    except statistics.StatisticsError:
        mode = 2
    if mode == 0:
        mode = 2
    return mode

# ===== Writers (giữ format txt/tsv/csv nhưng header động theo k) =====
def render_rows(rows: List[Dict[str,str]], k: int, fmt: str, labels: List[str]) -> str:
    headers = ["Mã số", "Tên chỉ tiêu"] + [labels[i] if i < len(labels) else f"Giá trị {i+1}" for i in range(k)]
    if fmt == "tsv":
        out = ["\t".join(headers)]
        for r in rows:
            row = [r.get("Mã số",""), r.get("Tên chỉ tiêu","")] + [r.get(f"Giá trị {i+1}","") for i in range(k)]
            out.append("\t".join(row))
        return "\n".join(out) + "\n"
    if fmt == "csv":
        out = [",".join(headers)]
        for r in rows:
            name = r.get("Tên chỉ tiêu","").replace(",", " ")
            row = [r.get("Mã số",""), name] + [r.get(f"Giá trị {i+1}","") for i in range(k)]
            out.append(",".join(row))
        return "\n".join(out) + "\n"

    # txt: căn lề
    w0 = max(len(headers[0]), max((len(r.get("Mã số","")) for r in rows), default=0))
    w1 = MAX_CHI_TIEU_LENGTH
    widths = []
    for i in range(k):
        col = labels[i] if i < len(labels) else f"Giá trị {i+1}"
        w = max(len(col), max((len(r.get(f"Giá trị {i+1}","")) for r in rows), default=0))
        widths.append(w)
    line = f"{headers[0]:<{w0}} | {headers[1]:<{w1}} | " + " | ".join(f"{(labels[i] if i < len(labels) else f'Giá trị {i+1}'):>{widths[i]}}" for i in range(k))
    sep  = f"{'-'*w0}-|{'-'*w1}-|-" + "-|-".join('-'*widths[i] for i in range(k))
    lines = [line, sep]
    for r in rows:
        values = " | ".join(f"{r.get(f'Giá trị {i+1}',''):>{widths[i]}}" for i in range(k))
        lines.append(f"{r.get('Mã số',''):<{w0}} | {r.get('Tên chỉ tiêu',''):<{w1}} | {values}")
    return "\n".join(lines) + "\n"

# ===== Prompt trùng file: y/n/a =====
def ask_overwrite_policy(path: str) -> str:
    print("\n[?] File đã tồn tại:", path)
    print("    y = overwrite (ghi đè), n = skip (bỏ qua), a = append (nối thêm)")
    while True:
        ans = input("    Nhập lựa chọn [y/n/a]: ").strip().lower()
        if ans in ("y", "n", "a"):
            return {"y":"overwrite","n":"skip","a":"append"}[ans]
        print("    Không hợp lệ, vui lòng nhập lại...")

def write_output(path: str, content: str, on_exist: str) -> bool:
    if os.path.exists(path):
        policy = on_exist
        if policy == "ask":
            policy = ask_overwrite_policy(path)
        if policy == "overwrite":
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return True
        elif policy == "skip":
            return False
        elif policy == "append":
            with open(path, "a", encoding="utf-8") as f:
                if content.count("\n") > 2 and path.lower().endswith("_table.txt"):
                    lines = content.splitlines()
                    tail = "\n".join(lines[2:]) + ("\n" if not content.endswith("\n") else "")
                    f.write(tail)
                else:
                    f.write(content)
            return True
        else:
            raise ValueError("Chính sách không hợp lệ: "+str(policy))
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return True

# ===== Xử lý 1 file =====
def process_one_file(in_path: str, out_root: str, in_root: str, fmt: str, on_exist: str,
                     fallback_only: bool, value_cols: Optional[int], labels: List[str],
                     allow_percent: bool, accept_small: bool) -> bool:
    rel = os.path.relpath(in_path, in_root)
    rel_dir = os.path.dirname(rel)
    base, _ = os.path.splitext(os.path.basename(rel))
    ext = {"txt":"_table.txt","tsv":"_table.tsv","csv":"_table.csv"}[fmt]
    out_dir = os.path.join(out_root, rel_dir)
    out_path = os.path.join(out_dir, base + ext)

    try:
        with open(in_path, "r", encoding="utf-8") as f:
            text = f.read()
    except UnicodeDecodeError:
        with open(in_path, "r", encoding="cp1258", errors="ignore") as f:
            text = f.read()

    lines = text.splitlines()
    k = value_cols or detect_value_cols(lines, allow_percent=allow_percent, accept_small=accept_small)
    k = max(1, min(4, k))

    rows: List[Dict[str, str]] = []
    if not fallback_only:
        for raw_line in lines:
            rec = parse_line(raw_line, k, allow_percent=allow_percent, accept_small=accept_small)
            if rec:
                rows.append(rec)

    if not rows:
        rows = parse_whole_text(text, k, allow_percent=allow_percent, accept_small=accept_small)

    if not rows:
        print(f"  [!] Bỏ qua (không có bản ghi hợp lệ): {in_path}")
        return False

    content = render_rows(rows, k, fmt, labels)
    wrote = write_output(out_path, content, on_exist)
    print(f"  [{'ĐÃ GHI' if wrote else 'BỎ QUA'}] {out_path}")
    return wrote

# ===== Main =====
def main():
    p = argparse.ArgumentParser("TXT → Table (UNIVERSAL, derived from base)")
    p.add_argument("--in", dest="in_dir", default=DEFAULT_IN_DIR, help="Thư mục đầu vào (mặc định đã điền)")
    p.add_argument("--out", dest="out_dir", default=DEFAULT_OUT_DIR, help="Thư mục đầu ra (mặc định đã điền)")
    p.add_argument("--fmt", choices=["txt","tsv","csv"], default=DEFAULT_FMT, help="Định dạng đầu ra")
    p.add_argument("--on-exist", choices=["ask","overwrite","skip","append"], default=DEFAULT_ON_EXIST,
                   help="Khi file đích đã tồn tại: hỏi/ghi đè/bỏ qua/nối thêm")
    p.add_argument("--fallback-only", action="store_true", help="Chỉ dùng bộ tách theo segment toàn văn bản (OCR 1 dòng dài)")
    p.add_argument("--value-cols", type=int, default=None, help="Cố định số cột giá trị ở cuối dòng (1..4). Mặc định: AUTO")
    p.add_argument("--labels", type=str, default="", help="CSV nhãn cho các cột giá trị, ví dụ: \"Số cuối năm,Số đầu năm\"")
    p.add_argument("--allow-percent", action="store_true", help="Nhận cả số % làm giá trị")
    p.add_argument("--accept-small", action="store_true", help="Nhận số nhỏ (<100k) để phù hợp bảng phi-tiền")
    args = p.parse_args()

    in_dir = os.path.abspath(args.in_dir)
    out_dir = os.path.abspath(args.out_dir)
    fmt = args.fmt
    on_exist = args.on_exist
    fallback_only = args.fallback_only
    value_cols = args.value_cols
    labels = [s.strip() for s in args.labels.split(",")] if args.labels else []
    allow_percent = args.allow_percent
    accept_small = args.accept_small

    if not os.path.isdir(in_dir):
        print("Lỗi: Không tìm thấy thư mục đầu vào:")
        print(f"  {in_dir}")
        print("Hãy kiểm tra lại đường dẫn DEFAULT_IN_DIR trong file hoặc truyền --in.")
        sys.exit(2)

    print("-"*80)
    print("BẮT ĐẦU CHUYỂN TXT → BẢNG (UNIVERSAL from base)")
    print(f"In : {in_dir}")
    print(f"Out: {out_dir}")
    print(f"Định dạng: {fmt.upper()} | Chính sách: {on_exist} | Fallback-only: {fallback_only}")
    print(f"Value-cols: {value_cols or 'AUTO'} | Labels: {labels or 'AUTO'} | %: {allow_percent} | accept-small: {accept_small}")
    print("-"*80)

    total, wrote = 0, 0
    for root, _, files in os.walk(in_dir):
        for name in files:
            if not name.lower().endswith(".txt"):
                continue
            total += 1
            in_path = os.path.join(root, name)
            try:
                if process_one_file(in_path, out_dir, in_dir, fmt, on_exist, fallback_only,
                                    value_cols, labels, allow_percent, accept_small):
                    wrote += 1
            except KeyboardInterrupt:
                print("\n^C Nhận Ctrl+C — dừng lại.")
                sys.exit(130)
            except Exception as e:
                print(f"  [LỖI] {in_path}: {e}")

    print("-"*80)
    print(f"Hoàn tất. Đã xử lý {total} file; trong đó {wrote} file được ghi ra '{out_dir}'.")
    print("-"*80)

if __name__ == "__main__":
    main()
