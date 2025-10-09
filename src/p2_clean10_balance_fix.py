# -*- coding: utf-8 -*-
"""
p2_clean10_balance_fix — Tự tìm *_text.txt mới nhất (5 cột) từ outputs/p1_clean10_orc_raw_output,
sửa số theo quan hệ cộng dồn và quy tắc tổng tài sản. Xuất ra:
  - *_numbers_fixed.txt (giữ 5 cột: Mã | Chỉ tiêu | Thuyết minh | Số cuối năm | Số đầu năm)
  - *_checks.txt (log)

Tuỳ chọn: đặt biến môi trường P2_FILE="...\\_pageX_text.txt" để ép chạy file chỉ định.
"""

from __future__ import annotations
import os, re, json, glob, unicodedata
from typing import Optional, List, Dict, Tuple

# --------- I/O roots ---------
IN_ROOT  = r"outputs/p1_clean10_orc_raw_output"
OUT_ROOT = r"outputs/p2_clean10_orc_raw_output"

# --------- Options ---------
FORCE_PARENT_SUM = True   # True: luôn ghi parent = tổng con nếu tổng con > 0

# ----------------- utils -----------------
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

_num_token = re.compile(r"(?<!\w)(?:\d[\d\s.,]{1,}\d)(?!\w)")
def clean_num(s: str) -> Optional[int]:
    """Chuẩn hoá chuỗi số và chuyển về int (chịu lỗi OCR)."""
    if not s:
        return None
    s = s.strip()
    # sửa nhầm lẫn OCR
    s = (s.replace("O", "0").replace("o", "0")
           .replace("I", "1").replace("l", "1")
           .replace("—", "-").replace("–", "-"))
    # lấy cụm số chính nếu có
    m = _num_token.search(s)
    s = m.group(0) if m else s
    # bỏ khoảng trắng, dấu phẩy
    s = s.replace(" ", "").replace(",", "")
    # bỏ dấu chấm ngăn nghìn (giữ âm nếu có)
    s = re.sub(r"\.(?=\d{3}(\D|$))", "", s)
    # bỏ ký tự không phải số/dấu âm
    s = re.sub(r"[^\d-]", "", s)
    if s in ("", "-", "--"):
        return None
    try:
        return int(s)
    except Exception:
        return None

def fmt_int(x: Optional[int]) -> str:
    return "" if x is None else f"{x:,}".replace(",", ".")

def _strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFD", str(text or ""))
    return "".join(ch for ch in text if not unicodedata.combining(ch))

def normalize_text(s: str) -> str:
    s = _strip_accents(s).lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ----------------- parse P1 (5 cột) -----------------
def _split5(line: str) -> Tuple[str,str,str,str,str]:
    parts = [p.strip() for p in line.split("|")]
    while len(parts) < 5:
        parts.append("")
    return parts[0], parts[1], parts[2], parts[3], parts[4]

_code_like = re.compile(r"^([0-9]{2,3}(?:\.[0-9]+)*)\b")

def parse_p1_line(line: str):
    """
    Mong đợi 5 cột: 'Mã số | Chỉ tiêu | Thuyết minh | Số cuối năm | Số đầu năm'
    - Nếu cột Mã số trống, thử bắt mã ở đầu 'Chỉ tiêu'
    """
    code_raw, desc, note, end_s, start_s = _split5(line)

    # bắt mã số từ cột 1; nếu rỗng, thử từ cột mô tả
    code = None
    if code_raw:
        m = _code_like.match(code_raw.replace(" ", ""))
        if m:
            code = m.group(1)

    if not code:
        m = _code_like.match((desc or "").replace(" ", ""))
        if m:
            code = m.group(1)

    end_v   = clean_num(end_s)
    start_v = clean_num(start_s)
    return code, desc, note, end_v, start_v

# ----------------- structure rules -----------------
def is_code(s: str) -> bool:
    return bool(re.fullmatch(r"[0-9]{2,3}(?:\.[0-9]+)*", s or ""))

def children_of(parent: str, all_codes: List[str]) -> List[str]:
    if not parent or not is_code(parent):
        return []
    kids = set()
    pfx_dot = parent + "."
    for c in all_codes:
        if c == parent:
            continue
        if c.startswith(pfx_dot):  # 151 -> 151.1, 151.2, 151.1.1...
            kids.add(c); continue
        # 120 -> 121..129 (+ 120.x)
        if re.fullmatch(r"[0-9]{3}", parent) and parent[-1] == "0" and parent[-2:] != "00":
            if re.fullmatch(r"[0-9]{3}", c) and c[:2] == parent[:2] and c[-1] != "0":
                kids.add(c); continue
        # 100 -> 110,120,...,190
        if re.fullmatch(r"[0-9]{3}", parent) and parent[-2:] == "00":
            if re.fullmatch(r"[0-9]{3}", c) and c[0] == parent[0] and c[1] != "0":
                kids.add(c); continue
    return sorted(kids, key=lambda x: [int(t) for t in x.split(".")])

KW_TOTAL_ASSETS   = ["tong cong tai san", "tong tai san", "total assets"]
KW_CUR_ASSETS     = ["tai san ngan han", "current assets"]
KW_NONCUR_ASSETS  = ["tai san dai han", "non current assets"]

def fix_numbers(rows: List[Dict], abs_tol: int = 5_000_000, rel_tol: float = 0.01):
    by_code = {r["code"]: r for r in rows if r.get("code")}
    all_codes = [r["code"] for r in rows if r.get("code")]
    checks = {"parents": [], "totals": []}

    def within_tol(a: Optional[int], b: Optional[int]) -> bool:
        if a is None or b is None:
            return False
        if abs(a - b) <= abs_tol:
            return True
        if max(abs(a), abs(b)) == 0:
            return a == b
        return abs(a - b) / max(abs(a), abs(b)) <= rel_tol

    # 1) parent = sum(children)
    for p in all_codes:
        kids = children_of(p, all_codes)
        if not kids:
            continue
        sum_end = sum(by_code[k]["end"] or 0 for k in kids if by_code[k]["end"] is not None)
        sum_start = sum(by_code[k]["start"] or 0 for k in kids if by_code[k]["start"] is not None)

        parent = by_code[p]
        orig_end, orig_start = parent["end"], parent["start"]
        new_end, new_start = orig_end, orig_start

        if FORCE_PARENT_SUM:
            if sum_end > 0:   new_end = sum_end
            if sum_start > 0: new_start = sum_start
        else:
            if orig_end is None or within_tol(orig_end, sum_end):
                new_end = sum_end if sum_end else orig_end
            if orig_start is None or within_tol(orig_start, sum_start):
                new_start = sum_start if sum_start else orig_start

        changed = (new_end != orig_end) or (new_start != orig_start)
        if changed:
            parent["end"], parent["start"] = new_end, new_start

        checks["parents"].append({
            "parent": p, "children": kids,
            "sum_end": sum_end, "sum_start": sum_start,
            "orig_end": orig_end, "orig_start": orig_start,
            "new_end": new_end, "new_start": new_start,
            "changed": changed
        })

    # 2) TOTAL ASSETS = CURRENT + NONCURRENT
    def find_by_kw(keywords: List[str]):
        for r in rows:
            if r["desc"] and any(kw in normalize_text(r["desc"]) for kw in keywords):
                return r
        return None

    total_row = find_by_kw(KW_TOTAL_ASSETS)
    cur_row   = find_by_kw(KW_CUR_ASSETS) or by_code.get("100")
    noncur_row= find_by_kw(KW_NONCUR_ASSETS) or by_code.get("200")

    if total_row and (cur_row or noncur_row):
        sum_end   = (cur_row["end"] if cur_row else 0) + (noncur_row["end"] if noncur_row else 0)
        sum_start = (cur_row["start"] if cur_row else 0) + (noncur_row["start"] if noncur_row else 0)
        orig_end, orig_start = total_row["end"], total_row["start"]
        new_end, new_start = orig_end, orig_start

        if FORCE_PARENT_SUM:
            if sum_end > 0:   new_end = sum_end
            if sum_start > 0: new_start = sum_start
        else:
            if orig_end is None or within_tol(orig_end, sum_end):
                new_end = sum_end if sum_end else orig_end
            if orig_start is None or within_tol(orig_start, sum_start):
                new_start = sum_start if sum_start else orig_start

        changed = (new_end != orig_end) or (new_start != orig_start)
        if changed:
            total_row["end"], total_row["start"] = new_end, new_start

        checks["totals"].append({
            "rule": "TOTAL_ASSETS = CURRENT + NONCURRENT",
            "cur": cur_row.get("code") if cur_row else None,
            "noncur": noncur_row.get("code") if noncur_row else None,
            "orig_end": orig_end, "orig_start": orig_start,
            "new_end": new_end, "new_start": new_start,
            "changed": changed
        })

    return rows, checks

# ----------------- I/O -----------------
def load_page(input_txt: str):
    base = os.path.basename(input_txt).replace("_text.txt", "")
    meta_path = input_txt.replace("_text.txt", "_meta.json")
    with open(input_txt, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]
    meta = {}
    if os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    return base, lines, meta, meta_path

def format_table_txt(rows: List[Dict]) -> str:
    col_code = max((len(r["code"] or "") for r in rows), default=2)
    col_desc = max((len(r["desc"] or "") for r in rows), default=8)
    col_note = max((len(r["note"] or "") for r in rows), default=2)

    header = (
        f"{'Mã số':<{col_code}} | "
        f"{'Chỉ tiêu':<{col_desc}} | "
        f"{'Thuyết':>{col_note}} | "
        f"{'Số cuối năm':>15} | {'Số đầu năm':>15}"
    )

    lines = [header, "-" * len(header)]
    for r in rows:
        code = (r["code"] or "").ljust(col_code)
        desc = (r["desc"] or "").ljust(col_desc)
        note = (r["note"] or "").rjust(col_note)
        end  = f"{fmt_int(r['end']):>15}"
        start= f"{fmt_int(r['start']):>15}"
        lines.append(f"{code} | {desc} | {note} | {end} | {start}")
    return "\n".join(lines)

def write_txt(path: str, content: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def mirror_out_dir(input_txt: str) -> str:
    in_root_norm = os.path.normpath(IN_ROOT)
    ipath_norm   = os.path.normpath(input_txt)
    if ipath_norm.startswith(in_root_norm):
        rel_dir = os.path.relpath(os.path.dirname(ipath_norm), in_root_norm)
    else:
        rel_dir = ""
    out_dir = os.path.join(OUT_ROOT, rel_dir)
    ensure_dir(out_dir)
    return out_dir

# ----------------- discover & run -----------------
def discover_latest_text() -> Optional[str]:
    env_path = os.environ.get("P2_FILE")
    if env_path and os.path.isfile(env_path):
        return env_path
    pattern = os.path.join(IN_ROOT, "**", "*_text.txt")
    candidates = glob.glob(pattern, recursive=True)
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]

def main():
    input_txt = discover_latest_text()
    if not input_txt:
        print(f"⚠️ Không tìm thấy *_text.txt trong {IN_ROOT}. "
              f"Bạn có thể đặt biến môi trường P2_FILE trỏ tới 1 file cụ thể.")
        return

    base, lines, meta, meta_path = load_page(input_txt)

    # parse (bỏ dòng rác không mã + không số)
    rows = []
    saw_first_code = False
    for ln in lines:
        if not ln.strip():
            continue

        code, desc, note, end_v, start_v = parse_p1_line(ln)

        # bỏ dòng rác không mã & không số
        if not code and end_v is None and start_v is None:
            continue

        if not saw_first_code and not code:
            # trước khi gặp dòng đầu có mã, các dòng có số nhưng không mã coi là rác → bỏ
            if end_v is None and start_v is None:
                continue
            else:
                continue

        if code:
            saw_first_code = True

        rows.append({"code": code, "desc": desc, "note": note, "end": end_v, "start": start_v})

    # fix
    fixed_rows, checks = fix_numbers(rows)

    # write
    out_dir   = mirror_out_dir(input_txt)
    fixed_txt = os.path.join(out_dir, f"{base}_numbers_fixed.txt")
    checks_txt= os.path.join(out_dir, f"{base}_checks.txt")

    write_txt(fixed_txt, format_table_txt(fixed_rows))

    log_lines = []
    for p in checks.get("parents", []):
        log_lines.append(
            f"[PARENT {p['parent']}] sum_end={fmt_int(p['sum_end'])} "
            f"sum_start={fmt_int(p['sum_start'])} "
            f"orig=({fmt_int(p['orig_end'])},{fmt_int(p['orig_start'])}) "
            f"new=({fmt_int(p['new_end'])},{fmt_int(p['new_start'])}) "
            f"changed={p['changed']}"
        )
    for t in checks.get("totals", []):
        log_lines.append(
            f"[TOTAL_ASSETS] orig=({fmt_int(t['orig_end'])},{fmt_int(t['orig_start'])}) "
            f"new=({fmt_int(t['new_end'])},{fmt_int(t['new_start'])}) "
            f"changed={t['changed']} cur={t['cur']} noncur={t['noncur']}"
        )
    write_txt(checks_txt, "\n".join(log_lines))

    print("✅ Input:", os.path.relpath(input_txt))
    if os.path.isfile(meta_path):
        print("ℹ️  Meta:", os.path.relpath(meta_path))
    print("📝 Out :", os.path.relpath(fixed_txt))
    print("🧾 Log :", os.path.relpath(checks_txt))

if __name__ == "__main__":
    main()
