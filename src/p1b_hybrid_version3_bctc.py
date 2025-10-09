"""
Step 1 – Hybrid Parser (version3)
File: p1b_hybrit_version3_bctc.py  (module name matches user's request)

Run (no args):
  python -m src.p1b_hybrit_version3_bctc

What it does by default
-----------------------
- Reads YAML at:    src/p1c_clean10_ocr_bctc.yaml
- Reads TXT folders:
    --dir-t392 => D:/1.TLAT/3. ChatBot_project/1_Insurance_Strategy - Copy/outputs/p1_clean10_version3_92
    --dir-bctc => D:/1.TLAT/3. ChatBot_project/1_Insurance_Strategy - Copy/outputs/p1a_clean10_ocr_bctc
- Writes outputs to:
    --outdir  => D:/1.TLAT/3. ChatBot_project/1_Insurance_Strategy - Copy/outputs/p1b_hybrid_version3_bctc
- Metadata defaults:
    company="UIC", period="2024-12-31", statement_type="balance_sheet.assets", unit="VND"
- Mode: ask (y=overwrite, n=skip, a=append) – one prompt for the whole batch
"""

from __future__ import annotations
import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yaml

# -----------------------------
# Defaults (per user's paths)
# -----------------------------
DEF_DIR_T392 = r"D:/1.TLAT/3. ChatBot_project/1_Insurance_Strategy - Copy/outputs/p1_clean10_version3_92"
DEF_DIR_BCTC = r"D:/1.TLAT/3. ChatBot_project/1_Insurance_Strategy - Copy/outputs/p1a_clean10_ocr_bctc"
DEF_OUTDIR   = r"D:/1.TLAT/3. ChatBot_project/1_Insurance_Strategy - Copy/outputs/p1b_hybrid_version3_bctc"
DEF_COMPANY  = "UIC"
DEF_PERIOD   = "2024-12-31"
DEF_STMT     = "balance_sheet.assets"
DEF_UNIT     = "VND"
DEF_MODE     = "ask"   # overwrite|skip|ask (ask => one-time prompt)

# YAML expected in same folder as this script
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEF_YAML = os.path.join(THIS_DIR, "p1c_clean10_ocr_bctc.yaml")

# -----------------------------
# YAML-driven parse config
# -----------------------------
@dataclass
class ParseConfig:
    unit: str = "VND"
    unit_multiplier: int = 1
    drop_chars: Tuple[str, ...] = (" ", ",", "£", "¥", "₫")
    fix_patterns: Tuple[Tuple[str, str], ...] = ()
    thousand_grouping: bool = True
    code_aliases: Dict[str, str] = None
    roman_normalize: Dict[str, str] = None

    @staticmethod
    def from_yaml(path: str, default_unit: str) -> "ParseConfig":
        with open(path, "r", encoding="utf-8") as f:
            y = yaml.safe_load(f)
        g = y.get("globals", {}) if isinstance(y, dict) else {}
        nc = (g.get("number_cleanup", {}) or {})
        unit = g.get("currency", default_unit or "VND")
        return ParseConfig(
            unit=unit,
            unit_multiplier=int(g.get("unit_multiplier", 1)),
            drop_chars=tuple(nc.get("drop_chars", [" ", ",", "£", "¥", "₫"])),
            fix_patterns=tuple((fp.get("from"), fp.get("to")) for fp in nc.get("fix_patterns", [])),
            thousand_grouping=bool(nc.get("thousand_grouping", True)),
            code_aliases=g.get("code_aliases", {}) or {},
            roman_normalize=g.get("roman_normalize", {}) or {},
        )

# -----------------------------
# Core regex & cleaners
# -----------------------------
_CODE_AT_START = re.compile(r"^\s*(?P<code>\d{3}(?:[.\s]\d{1,2})?)\b")

_NUMBER_CAPTURE = re.compile(r"([\(\-]?[0-9OIl\.,\s]+[)]?)")


def extract_amount_tokens(text: str, cfg: ParseConfig) -> List[str]:
    """
    Lấy các token số có khả năng là số tiền, loại bỏ số 'thuyết minh' (4, 5.2, 7...).
    Điều kiện: sau khi clean, có >= 6 chữ số.
    """
    cand = _NUMBER_CAPTURE.findall(text)
    out: List[str] = []
    for tok in cand:
        t0 = (tok.replace("O","0").replace("o","0").replace("I","1").replace("l","1")
                .replace("–","-").strip())
        if t0.startswith("(") and t0.endswith(")"):
            t0 = t0[1:-1]
        for ch in cfg.drop_chars:
            t0 = t0.replace(ch, "")
        t1 = re.sub(r"[^0-9\.,-]", "", t0)
        digits = re.sub(r"[^0-9]", "", t1)
        if len(digits) >= 7:
            out.append(tok)
    return out


def _apply_fix_patterns(s: str, patterns: Tuple[Tuple[str, str], ...]) -> str:
    for frm, to in patterns:
        if not frm:
            continue
        s = re.sub(frm, to, s)
    return s

def normalize_code(raw: str, aliases: Dict[str, str]) -> str:
    raw = (raw or "").strip()
    raw = raw.replace(" ", ".")  # '131 1' -> '131.1'
    if raw in aliases:
        return aliases[raw]
    m = re.match(r"^(\d{3})(\d{1,2})$", raw)
    if m:
        cand = f"{m.group(1)}.{m.group(2)}"
        return aliases.get(cand, cand)
    return raw


def clean_number(raw: Optional[str], cfg: ParseConfig) -> Optional[int]:
    if raw is None:
        return None
    s = str(raw)
    s = (s.replace("O","0").replace("o","0").replace("I","1").replace("l","1")
           .replace("–","-").strip())
    s = _apply_fix_patterns(s, cfg.fix_patterns)
    neg = s.startswith("(") and s.endswith(")")
    if neg:
        s = s[1:-1]

    for ch in cfg.drop_chars:
        s = s.replace(ch, "")

    if s.count(".") + s.count(",") >= 2:
        s = re.sub(r"[^0-9\-]", "", s)
    else:
        s = re.sub(r"[^0-9\.\-]", "", s)

    if cfg.thousand_grouping and s.count(".") > 0 and "," not in s:
        s = s.replace(".", "")

    if s.endswith("-"):
        s = s[:-1]
    if not s or s in {"-", "."}:
        return None

    try:
        val = int(float(s))
    except ValueError:
        return None
    if neg:
        val = -val
    return val * int(cfg.unit_multiplier)


# -----------------------------
# ASCII table (version3-92) parsing
# -----------------------------
def is_ascii_table(text: str) -> bool:
    lines = text.splitlines()
    pipe_lines = sum(1 for ln in lines if '|' in ln)
    return pipe_lines >= max(6, len(lines)//4)

def parse_ascii_table(text: str, cfg: ParseConfig) -> List[Dict[str, Optional[str]]]:
    def is_sep_cells(cells: List[str]) -> bool:
        # tất cả ô chỉ gồm -+_ hoặc trống => dòng kẻ
        return all(set(c) <= set("-+_ ") for c in cells)

    def is_money_like(tok: str) -> bool:
        # token có >=6 chữ số sau khi làm sạch -> coi là số tiền
        t = re.sub(r"[^0-9]", "", tok or "")
        return len(t) >= 6

    lines = text.splitlines()
    rows: List[Dict[str, Optional[str]]] = []
    i = 0
    while i < len(lines):
        ln = lines[i].strip()
        i += 1
        if "|" not in ln:
            continue

        parts = [c.strip() for c in ln.split("|")]
        if len(parts) < 6:
            continue

        code_cell  = parts[1]
        label_cell = parts[2]
        note_cell  = parts[3]
        end_cell   = parts[4]
        begin_cell = parts[5]

        # bỏ header, dòng kẻ
        low = code_cell.lower()
        if low in {"mã số", "ma so", "ms"}:
            continue
        if is_sep_cells([code_cell, label_cell, note_cell, end_cell, begin_cell]):
            continue

        # dòng tiếp diễn label (không có mã)
        if not code_cell:
            if rows:
                rows[-1]["label"] = (rows[-1].get("label") or "") + " " + (label_cell or "")
                rows[-1]["label"] = rows[-1]["label"].strip()
            continue

        # nếu 2 cột tiền chưa có / quá nhỏ -> thử nhìn dòng kế
        if not (is_money_like(end_cell) or is_money_like(begin_cell)):
            if i < len(lines) and "|" in lines[i]:
                parts_next = [c.strip() for c in lines[i].strip().split("|")]
                if len(parts_next) >= 6:
                    end_n, begin_n = parts_next[4], parts_next[5]
                    if is_money_like(end_n) or is_money_like(begin_n):
                        end_cell = end_n or end_cell
                        begin_cell = begin_n or begin_cell
                        i += 1  # đã dùng dòng kế tiếp cho số tiền

        code_norm = normalize_code(code_cell, cfg.code_aliases)
        rows.append({
            "raw_code": code_cell,
            "code": code_norm,
            "label": label_cell or None,
            "note":  note_cell or None,
            "end_raw": end_cell or None,
            "begin_raw": begin_cell or None,
        })

    return rows


# -----------------------------
# Free text (BCTC) line parser
# -----------------------------
def split_line_to_row(line: str, cfg: ParseConfig) -> Optional[Dict[str, Optional[str]]]:
    line = line.rstrip("\n").strip()
    if not line:
        return None
    m = _CODE_AT_START.search(line)
    if not m:
        return None
    raw_code = m.group("code")
    rest = line[m.end():].strip()

    # chỉ lấy những token "đủ lớn" (lọc số thuyết minh)
    nums = extract_amount_tokens(rest, cfg)

    end_raw = begin_raw = None
    label = rest
    if nums:
        if len(nums) >= 2:
            begin_raw = nums[-1]
            end_raw   = nums[-2]
            first_num_pos = rest.rfind(nums[-2])
        else:
            end_raw = nums[-1]
            first_num_pos = rest.rfind(nums[-1])
        if first_num_pos != -1:
            label = rest[:first_num_pos].strip()

    for k, v in (cfg.roman_normalize or {}).items():
        label = label.replace(k, v)

    code = normalize_code(raw_code, cfg.code_aliases)
    return {
        "raw_code": raw_code,
        "code": code,
        "label": label,
        "note": None,
        "end_raw": end_raw,
        "begin_raw": begin_raw,
    }


def parse_text_to_rows(text: str, cfg: ParseConfig) -> List[Dict[str, Optional[str]]]:
    rows: List[Dict[str, Optional[str]]] = []
    lines = text.splitlines()
    i = 0
    buffer: Optional[Dict[str, Optional[str]]] = None

    while i < len(lines):
        raw = lines[i]
        row = split_line_to_row(raw, cfg)

        if row is None:
            # continuation của label cho dòng trước
            if buffer is not None:
                cont = raw.strip()
                if cont:
                    buffer["label"] = (buffer.get("label") or "").rstrip() + " " + cont
            i += 1
            continue

        # nếu chưa bắt được số ở dòng hiện tại → thử look-ahead 1 dòng
        if row["end_raw"] is None and row["begin_raw"] is None and (i + 1) < len(lines):
            nxt = lines[i + 1]
            extra = extract_amount_tokens(nxt, cfg)
            if extra:
                if len(extra) >= 2:
                    row["end_raw"], row["begin_raw"] = extra[-2], extra[-1]
                else:
                    row["end_raw"] = extra[-1]
                i += 1  # đã dùng dòng kế, nhảy qua để khỏi dính vào nhãn dòng sau

        if buffer is not None:
            rows.append(buffer)
        buffer = row
        i += 1

    if buffer is not None:
        rows.append(buffer)
    return rows


# -----------------------------
# Dataframe builder
# -----------------------------
def rows_to_dataframe(
    rows: List[Dict[str, Optional[str]]],
    meta: Dict[str, str],
    cfg: ParseConfig,
    source: str,
    file_path: str
) -> pd.DataFrame:
    records = []
    for r in rows:
        end = clean_number(r.get("end_raw"), cfg)
        begin = clean_number(r.get("begin_raw"), cfg)
        records.append({
            **meta,
            "code": r.get("code"),
            "label": (r.get("label") or "").strip() or None,
            "note": r.get("note"),
            "end": end,
            "begin": begin,
            "source": source,
            "file": os.path.basename(file_path),
            "raw_code": r.get("raw_code"),
            "end_raw": r.get("end_raw"),
            "begin_raw": r.get("begin_raw"),
        })
    return pd.DataFrame.from_records(records, columns=[
        "company","period","statement_type","unit",
        "code","label","note","end","begin","source","file",
        "raw_code","end_raw","begin_raw"
    ])

# -----------------------------
# IO helpers
# -----------------------------
def list_txt(dir_path: str, recursive: bool = True) -> List[str]:
    exts = {".txt", ".TXT", ".text"}
    files: List[str] = []
    if not os.path.isdir(dir_path):
        return files
    if recursive:
        for root, _, fnames in os.walk(dir_path):
            for f in fnames:
                _, ext = os.path.splitext(f)
                if ext in exts:
                    files.append(os.path.join(root, f))
    else:
        for f in os.listdir(dir_path):
            _, ext = os.path.splitext(f)
            if ext in exts:
                files.append(os.path.join(dir_path, f))
    return files

def parse_dir(
    dir_path: str,
    meta: Dict[str, str],
    cfg: ParseConfig,
    source: str,
    debug: bool = True
) -> pd.DataFrame:
    files = list_txt(dir_path, recursive=True)
    if debug:
        print(f"[{source}] Found {len(files)} text files under: {dir_path}")
        for p in files[:5]:
            print(f"  · {p}")
    dfs: List[pd.DataFrame] = []
    for p in files:
        text: Optional[str] = None
        for enc in ("utf-8", "utf-8-sig", "utf-16", "utf-16le", "utf-16be", "cp1258", "latin-1"):
            try:
                with open(p, "r", encoding=enc, errors="strict") as f:
                    text = f.read()
                    break
            except Exception:
                continue
        if text is None:
            print(f"[WARN] Unable to read file with known encodings: {p}")
            continue

        # Choose parser by layout
        if is_ascii_table(text):
            rows = parse_ascii_table(text, cfg)     # version3-92 table
        else:
            rows = parse_text_to_rows(text, cfg)    # free-text BCTC

        if debug and not rows:
            print(f"[WARN] No rows parsed from: {os.path.basename(p)}")
            preview = "\n".join(text.splitlines()[:3])
            print(f"  Preview:\n{preview}")

        df = rows_to_dataframe(rows, meta, cfg, source=source, file_path=p)
        dfs.append(df)

    if not dfs:
        return pd.DataFrame(columns=[
            "company","period","statement_type","unit",
            "code","label","note","end","begin","source","file",
            "raw_code","end_raw","begin_raw"
        ])
    return pd.concat(dfs, ignore_index=True)

# -----------------------------
# Reporting helpers
# -----------------------------
def quick_checks(df: pd.DataFrame) -> Dict[str, object]:
    total = int(len(df)) if df is not None else 0
    if total == 0:
        return {
            "rows": 0,
            "unique_codes": 0,
            "duplicate_codes": {},
            "%_with_end_or_begin": 0.0,
            "sample_head": [],
        }
    uniq_codes = int(df["code"].nunique(dropna=True))
    dup_series = df["code"].value_counts()
    dup_codes = dup_series[dup_series > 1].to_dict()
    has_number = int(((df["end"].notna()) | (df["begin"].notna())).sum())
    pct_with_number = round(100.0 * has_number / total, 2)
    return {
        "rows": total,
        "unique_codes": uniq_codes,
        "duplicate_codes": dup_codes,
        "%_with_end_or_begin": pct_with_number,
        "sample_head": df.head(5).to_dict(orient="records"),
    }

# -----------------------------
# CLI (with defaults)
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Hybrid Step 1: parse folders to normalized CSVs (version3)")
    ap.add_argument("--dir-t392", default=DEF_DIR_T392, help="Folder of version3-92 TXT files")
    ap.add_argument("--dir-bctc", default=DEF_DIR_BCTC, help="Folder of BCTC TXT files")
    ap.add_argument("--yaml", default=DEF_YAML, help="Path to YAML (globals used)")
    ap.add_argument("--outdir", default=DEF_OUTDIR, help="Output directory")
    ap.add_argument("--company", default=DEF_COMPANY)
    ap.add_argument("--period", default=DEF_PERIOD)
    ap.add_argument("--statement-type", default=DEF_STMT)
    ap.add_argument("--unit", default=DEF_UNIT, help="Override unit (default from YAML globals.currency)")
    ap.add_argument("--mode", choices=["overwrite","skip","ask"], default=DEF_MODE)

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    out_bctc = os.path.join(args.outdir, "parsed_bctc.csv")
    out_t392 = os.path.join(args.outdir, "parsed_3_92.csv")
    out_report = os.path.join(args.outdir, "parse_report.json")

    existing = [p for p in [out_bctc, out_t392, out_report] if os.path.exists(p)]
    mode = args.mode
    if existing and args.mode == "ask":
        print("Các file đầu ra đã tồn tại:\n - " + "\n - ".join(existing))
        ans = input("Ghi đè tất cả? (y = overwrite, n = skip, a = append) : ").strip().lower()
        if ans == "y":
            mode = "overwrite"
        elif ans == "n":
            mode = "skip"
        else:
            mode = "append"

    cfg = ParseConfig.from_yaml(args.yaml, default_unit=args.unit)
    if args.unit:
        cfg.unit = args.unit

    meta = {
        "company": args.company,
        "period": args.period,
        "statement_type": args.__dict__["statement_type"],
        "unit": cfg.unit,
    }

    # parse both dirs
    df_t392 = parse_dir(args.dir_t392, meta, cfg, source="3_92_txt")
    df_bctc = parse_dir(args.dir_bctc, meta, cfg, source="bctc_txt")

    # handle write mode
    if os.path.exists(out_bctc) and mode == "skip":
        print("SKIP ghi file parsed_bctc.csv (đã tồn tại)")
    elif os.path.exists(out_bctc) and mode == "append":
        old = pd.read_csv(out_bctc, dtype=str)
        new = pd.concat([old, df_bctc], ignore_index=True)
        new.to_csv(out_bctc, index=False, encoding="utf-8-sig")
    else:
        df_bctc.to_csv(out_bctc, index=False, encoding="utf-8-sig")

    if os.path.exists(out_t392) and mode == "skip":
        print("SKIP ghi file parsed_3_92.csv (đã tồn tại)")
    elif os.path.exists(out_t392) and mode == "append":
        old = pd.read_csv(out_t392, dtype=str)
        new = pd.concat([old, df_t392], ignore_index=True)
        new.to_csv(out_t392, index=False, encoding="utf-8-sig")
    else:
        df_t392.to_csv(out_t392, index=False, encoding="utf-8-sig")

    report = {
        "bctc": quick_checks(df_bctc),
        "t392": quick_checks(df_t392),
        "config": {
            "unit": cfg.unit,
            "unit_multiplier": cfg.unit_multiplier,
            "drop_chars": cfg.drop_chars,
            "thousand_grouping": cfg.thousand_grouping,
        },
        "paths": {"bctc_csv": out_bctc, "t392_csv": out_t392},
        "inputs": {"dir_t392": args.dir_t392, "dir_bctc": args.dir_bctc, "yaml": args.yaml},
        "mode": mode,
    }

    with open(out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
