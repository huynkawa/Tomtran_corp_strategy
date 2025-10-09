"""
P1B – Parse OCR TXT -> structured table for BCTC (code | name | note | end | begin)

Usage (ngắn, không hỏi gì):
    python -m src.p1b_clean10_ocr_bctc

Mặc định:
  - Input : D:\1.TLAT\3. ChatBot_project\1_Insurance_Strategy - Copy\outputs\p1a_clean10_ocr_bctc
  - Output: D:\1.TLAT\3. ChatBot_project\1_Insurance_Strategy - Copy\outputs\p1b_clean10_ocr_bctc
  - Format: TXT (bảng cố định cột)
  - Mode  : a (ghi đè nếu trùng)

Có thể đổi bằng tham số:
  --out_fmt txt|csv   ;  --mode y|n|a
"""
from __future__ import annotations
import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

# ------------------------- Config defaults -------------------------
DEFAULT_IN = r"D:\\1.TLAT\\3. ChatBot_project\\1_Insurance_Strategy - Copy\\outputs\\p1a_clean10_ocr_bctc"
DEFAULT_OUT = r"D:\\1.TLAT\\3. ChatBot_project\\1_Insurance_Strategy - Copy\\outputs\\p1b_clean10_ocr_bctc_TXT"

# match codes like 100, 131.1, 151.2 at start of line (allow some leading spaces)
RE_CODE = re.compile(r"(?<![\d.])\b([0-9]{3}(?:\.[0-9])?)\b")
# capture large numeric tokens (amounts) with thousands separators
RE_NUM = re.compile(r"-?\d[\d.,]{5,}")
# capture small note tokens (1..2 digits or like 5.2)
RE_NOTE = re.compile(r"(?<!\d)(\d{1,2}(?:[.,]\d)?)(?!\d)")


@dataclass
class OutConfig:
    fmt: str = "txt"   # "txt" | "csv"  (DEFAULT txt)
    mode: str = "a"    # y=delete all old, n=skip existing, a=overwrite/add


def clean_amount_token(tok: Optional[str]) -> Optional[int]:
    if not tok:
        return None
    s = tok.strip().replace(",", "").replace(".", "")
    m = re.findall(r"-?\d+", s)
    if not m:
        return None
    try:
        return int("".join(m))
    except Exception:
        return None


def pick_two_amounts(line: str) -> Tuple[Optional[str], Optional[str]]:
    """Pick two longest numeric tokens by digit-length; preserve order.
    Returns (end_raw, begin_raw).
    """
    nums = list(RE_NUM.finditer(line))
    if not nums:
        return (None, None)
    scored = []
    for m in nums:
        tok = m.group(0)
        digits = re.sub(r"[^0-9]", "", tok)
        scored.append((m.start(), tok, len(digits)))
    top2 = sorted(scored, key=lambda x: (-x[2], x[0]))[:2]
    top2_sorted = [t[1] for t in sorted(top2, key=lambda x: x[0])]
    if len(top2_sorted) == 1:
        return (top2_sorted[0], None)
    return (top2_sorted[0], top2_sorted[1])


def extract_note(line: str) -> Optional[str]:
    candidates = []
    for m in RE_NOTE.finditer(line):
        tok = m.group(1)
        digits_only = re.sub(r"[^0-9]", "", tok)
        if len(digits_only) <= 2 or re.match(r"^\d{1,2}[.,]\d$", tok):
            candidates.append((m.start(), tok))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1].replace(",", ".")


def clean_name(raw_line: str, code: Optional[str], note: Optional[str], end_raw: Optional[str], begin_raw: Optional[str]) -> str:
    s = raw_line
    if code:
        s = re.sub(rf"^\s*{re.escape(code)}\b\s*", "", s)
    for tok in (end_raw, begin_raw):
        if tok:
            s = s.replace(tok, " ")
    if note:
        s = re.sub(rf"(?<!\d){re.escape(note)}(?!\d)", " ", s)
    s = re.sub(r"\s+", " ", s).strip(" -:;|\t\r\n")
    return s


def parse_txt_lines(lines: List[str]) -> List[dict]:
    rows = []
    for line in lines:
        raw = line.rstrip("\n")
        if not raw.strip():
            continue
        mcode = RE_CODE.search(raw)
        code = mcode.group(1) if mcode else None
        end_raw, begin_raw = pick_two_amounts(raw)
        note = extract_note(raw)
        name = clean_name(raw, code, note, end_raw, begin_raw)
        end_val = clean_amount_token(end_raw)
        begin_val = clean_amount_token(begin_raw)
        if code or (end_val is not None) or (begin_val is not None):
            rows.append({
                "code": code,
                "name": name,
                "note": note,
                "end": end_val,
                "begin": begin_val,
                "raw_line": raw,
                "end_raw": end_raw,
                "begin_raw": begin_raw,
            })
    return rows


def read_txt_file(path: Path) -> List[str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.readlines()


def write_csv(out_path: Path, rows: List[dict]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["code","name","note","end","begin","raw_line","end_raw","begin_raw"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_txt_table(out_path: Path, rows: List[dict]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["code","name","note","end","begin"]
    def val_str(v: Optional[object]) -> str:
        return "" if v is None else str(v)
    widths = {h: len(h) for h in headers}
    for r in rows:
        for h in headers:
            widths[h] = max(widths[h], len(val_str(r.get(h))))
    def pad(s: str, w: int) -> str:
        return s + " " * (w - len(s))
    line_sep = "+" + "+".join(["-" * (widths[h] + 2) for h in headers]) + "+"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(line_sep + "\n")
        f.write("| " + " | ".join([pad(h, widths[h]) for h in headers]) + " |\n")
        f.write(line_sep + "\n")
        for r in rows:
            vals = [val_str(r.get(h)) for h in headers]
            f.write("| " + " | ".join([pad(vals[i], widths[headers[i]]) for i in range(len(headers))]) + " |\n")
        f.write(line_sep + "\n")


def list_inputs(in_dir: Path) -> List[Path]:
    return sorted(in_dir.glob("*.txt"))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="P1B parser: TXT -> structured TXT/CSV table for BCTC")
    parser.add_argument("--in_dir", default=DEFAULT_IN, help="Input folder that contains *.txt (and meta.json)")
    parser.add_argument("--out_dir", default=DEFAULT_OUT, help="Output folder for parsed files")
    parser.add_argument("--mode", choices=["y","n","a"], default="a", help="y=delete old, n=skip existing, a=overwrite/add (default)")
    parser.add_argument("--out_fmt", choices=["txt","csv"], default="txt", help="Output format: txt (default) or csv")
    args = parser.parse_args(argv)

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    txt_files = list_inputs(in_dir)
    if not txt_files:
        print(f"No .txt files found in {in_dir}")
        return 1

    out_ext = ".txt" if args.out_fmt == "txt" else ".csv"

    # No prompt (ngắn gọn). If mode==y, clear matching outputs first
    if args.mode == "y":
        deleted = 0
        for p in out_dir.glob(f"*{out_ext}"):
            try:
                p.unlink()
                deleted += 1
            except Exception:
                pass
        print(f"[mode=y] Deleted {deleted} old {out_ext} files from {out_dir}")

    written = skipped = failed = 0

    for txt_path in txt_files:
        try:
            base = txt_path.stem
            out_file = out_dir / f"{base}{out_ext}"
            if args.mode == "n" and out_file.exists():
                skipped += 1
                continue
            lines = read_txt_file(txt_path)
            rows = parse_txt_lines(lines)
            if args.out_fmt == "txt":
                write_txt_table(out_file, rows)
            else:
                write_csv(out_file, rows)
            written += 1
        except Exception as e:
            failed += 1
            print(f"[ERROR] {txt_path.name}: {e}")

    print(f"Done. written={written}, skipped={skipped}, failed={failed}. Output: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
