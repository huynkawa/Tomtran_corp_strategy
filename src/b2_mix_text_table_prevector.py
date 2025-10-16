# -*- coding: utf-8 -*-
"""
src.b2_mix_text_table_prevector.py
c2_clean_tables_pre_vector.py — BẢN CHUYÊN CHO BẢNG
- Quét cả thư mục input (mirror sang out-root)
- Chỉ xử lý các block [TABLE ...] trong *_text.txt|*_text.final.txt
- Clean TSV mạnh tay + xuất <stem>_tables.final.tsv và <stem>_vector.jsonl

Chạy ví dụ:
  python -m src.c2_clean_tables_pre_vector
  # hoặc chỉ xử lý file mới:
  python -m src.c2_clean_tables_pre_vector --skip-existing
"""

import os, re, json, argparse
from pathlib import Path
from typing import List, Tuple, Optional
from collections import Counter

IN_ROOT_DEFAULT  = r"D:\1.TLAT\3. ChatBot_project\1_Insurance_Strategy\outputs\b1_mix_text_table_output"
OUT_ROOT_DEFAULT = r"D:\1.TLAT\3. ChatBot_project\1_Insurance_Strategy\outputs\b2_mix_text_table_prevector"
PATTERN_DEFAULT  = "*_text.txt|*_text.final.txt"

HDR_BRACKETS = re.compile(r"\[([^\]]+)\]")
IS_HEADER    = re.compile(r"^###\s*(\[[^\]]+\]\s*)+", re.I)

def parse_header(line: str) -> dict:
    groups = re.findall(HDR_BRACKETS, line)
    meta = {
        "doc_type": None, "page": None, "sheet": None,
        "table_index": None, "content_type_hint": None,
        "raw_header": line.strip()
    }
    for g in groups:
        g_up = g.upper()
        if "TABLE" in g_up:
            meta["content_type_hint"] = "TABLE"
            m = re.search(r"TABLE\s+(\d+)", g_up)
            if m: meta["table_index"] = int(m.group(1))
            continue
        if "TEXT" in g_up:
            meta["content_type_hint"] = "TEXT"; continue
        if g_up.startswith("PDF"):
            meta["doc_type"] = "PDF"
            m = re.search(r"PAGE\s+(\d+)", g_up)
            if m: meta["page"] = int(m.group(1)); continue
        if g_up.startswith("DOCX"):  meta["doc_type"] = "DOCX";  continue
        if g_up.startswith("IMG"):   meta["doc_type"] = "IMG";   continue
        if g_up.startswith("TXT"):   meta["doc_type"] = "TXT";   continue
        if g_up.startswith("EXCEL"): meta["doc_type"] = "EXCEL"; continue
        if g_up.startswith("SHEET="): meta["sheet"] = g.split("=",1)[-1]; continue
    return meta

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_base_meta(input_txt_path: Path) -> dict:
    meta_path = Path(str(input_txt_path).replace("_text.final.txt", "_meta.json"))
    if not meta_path.exists():
        meta_path = Path(str(input_txt_path).replace("_text.txt", "_meta.json"))
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

# ===== Các quy tắc clean cho TABLE =====
NOISY_ROW_RE = re.compile(
    r"^(GNỘĐCÁTỰS|KHẢ NĂNG XẢY RA\b|\d+\s*-\s*Không\s*$|^Trang\s*\d+\s*$|^Page\s*\d+\s*$)",
    re.I
)

def split_tsv_line(raw: str) -> List[str]:
    return [c.strip() for c in raw.split("\t")]

def join_tsv_row(cells: List[str]) -> str:
    return "\t".join((c or "").strip() for c in cells)

def most_common_cols(lines: List[str]) -> int:
    counts = [len(split_tsv_line(ln)) for ln in lines if ln.strip()]
    if not counts: return 0
    return Counter(counts).most_common(1)[0][0]

def fix_common_table_row(raw: str) -> str:
    # sửa lỗi OCR/đánh máy nhẹ
    raw = raw.replace("", "★")
    raw = re.sub(r"xảy\s*xa\b", "xảy ra", raw, flags=re.I)
    # gộp CHỮ IN BỊ CÁCH KÝ TỰ: "K H Ả N Ă N G" -> "KHẢ NĂNG"
    raw = re.sub(r'((?:[A-ZÀ-ỸĐ]\s+){3,}[A-ZÀ-ỸĐ])', lambda m: m.group(0).replace(" ", ""), raw)
    return raw

def clean_table_block_strong(
    lines: List[str],
    min_cols: int,
    max_cols: int,
    fix_rows: bool = True,
    target_cols: int = 0
) -> List[str]:
    # 1) lọc rác cơ bản & tách cells
    rows_cells: List[List[str]] = []
    filtered: List[str] = []
    for raw in lines:
        raw = fix_common_table_row(raw)
        if not raw.strip():           continue
        if NOISY_ROW_RE.match(raw):   continue
        filtered.append(raw)

    if not filtered:
        return []

    # 2) xác định số cột chuẩn
    dom = target_cols if target_cols > 0 else most_common_cols(filtered)
    if dom == 0: dom = min_cols

    # 3) nạp hàng, hợp nhất hàng thiếu cột vào ô cuối của hàng trước (wrap)
    for raw in filtered:
        cells = split_tsv_line(raw)
        if all(c == "" for c in cells):    continue
        if len(cells) < min_cols:          continue

        if fix_rows and rows_cells and 0 < len(cells) < dom:
            # nối vào cell cuối của hàng trước
            prev = rows_cells[-1]
            prev[-1] = (prev[-1] + " " + " ".join(cells)).strip()
            continue

        # chuẩn hóa số cột
        if len(cells) > max_cols:
            cells = cells[:max_cols]
        if dom and len(cells) < dom:
            cells = cells + [""] * (dom - len(cells))

        rows_cells.append([re.sub(r"\s+", " ", c) for c in cells])

    # 4) loại header lặp lại (trùng y hệt cách nhau ≥2 hàng)
    seen = set()
    cleaned_rows: List[List[str]] = []
    for i, row in enumerate(rows_cells):
        key = tuple(row)
        if key in seen:
            # bỏ lặp rõ ràng
            continue
        cleaned_rows.append(row)
        if i <= 5:  # chỉ add vào seen cho vài hàng đầu (thường là header)
            seen.add(key)

    # 5) xuất
    out_lines = []
    for r in cleaned_rows:
        if len(r) < min_cols:  # safety
            continue
        out_lines.append(join_tsv_row(r))
    return out_lines

# ===== Walk & process =====
def walk_text_files(in_root: Path, pattern: str) -> List[Path]:
    pats = [p.strip() for p in (pattern or "*_text.txt").split("|") if p.strip()]
    found = []
    for root, _, files in os.walk(in_root):
        for fn in files:
            for pat in pats:
                if Path(fn).match(pat):
                    found.append(Path(root) / fn); break
    return found

def process_one_file(
    input_txt: Path,
    in_root: Path,
    out_root: Path,
    skip_existing: bool,
    min_cols: int,
    max_cols: int,
    target_cols: int,
    no_merge_rows: bool
) -> Tuple[str, Path, Optional[Path]]:
    base_meta = load_base_meta(input_txt)

    rel_dir = input_txt.parent.relative_to(in_root)
    out_dir = out_root / rel_dir
    ensure_dir(out_dir)

    stem_in = input_txt.name.replace("_text.final.txt","").replace("_text.txt","")
    out_tsv = out_dir / f"{stem_in}_tables.final.tsv"
    out_jsonl = out_dir / f"{stem_in}_vector.jsonl"

    if skip_existing and out_tsv.exists() and out_jsonl.exists():
        return ("skip", out_tsv, out_jsonl)

    raw_lines = input_txt.read_text(encoding="utf-8", errors="ignore").splitlines()

    # gom riêng các block TABLE
    tables: List[Tuple[dict, List[str]]] = []
    cur_hdr = None
    cur_buf: List[str] = []

    def flush():
        nonlocal cur_hdr, cur_buf
        if cur_hdr and (cur_hdr.get("content_type_hint") or "").upper() == "TABLE":
            tables.append((cur_hdr, cur_buf[:]))
        cur_hdr = None; cur_buf = []

    for ln in raw_lines:
        if IS_HEADER.match(ln):
            # gặp header mới
            hdr = parse_header(ln)
            # xả block trước
            flush()
            cur_hdr = hdr
        else:
            cur_buf.append(ln)
    flush()

    # làm sạch & ghi
    final_tsv_lines = []
    jsonl_records = []
    for hdr, buf in tables:
        lines = clean_table_block_strong(
            buf, min_cols=min_cols, max_cols=max_cols,
            fix_rows=(not no_merge_rows),
            target_cols=target_cols
        )
        if not lines:
            continue
        # đánh dấu block trong tsv tổng
        final_tsv_lines.append(hdr.get("raw_header","### [TABLE]"))
        final_tsv_lines.extend(lines)

        # jsonl: mỗi bảng là 1 record (gộp toàn bộ block)
        content = "\n".join(lines).strip()
        md = {k:v for k,v in hdr.items() if v is not None}
        for k,v in base_meta.items():
            if k == "text_sha1":  # bỏ hash cũ
                continue
            if k not in md:
                md[k] = v
        md["source_output"] = out_tsv.name
        jsonl_records.append({"content": content, "metadata": md, "content_type": "TABLE"})

    # ghi file
    out_tsv.write_text("\n".join(final_tsv_lines).strip() + ("\n" if final_tsv_lines else ""), encoding="utf-8")
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in jsonl_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return ("done", out_tsv, out_jsonl)

def main():
    ap = argparse.ArgumentParser(description="Clean TABLES only & export TSV + JSONL (mirror tree).")
    ap.add_argument("--in-root",  default=IN_ROOT_DEFAULT)
    ap.add_argument("--out-root", default=OUT_ROOT_DEFAULT)
    ap.add_argument("--pattern",  default=PATTERN_DEFAULT, help="*_text.txt|*_text.final.txt")
    ap.add_argument("--skip-existing", action="store_true", help="Bỏ qua file đã có _tables.final.tsv và _vector.jsonl")
    ap.add_argument("--min-cols", type=int, default=1)
    ap.add_argument("--max-cols", type=int, default=1000)
    ap.add_argument("--target-cols", type=int, default=0, help="Nếu >0, ép số cột chuẩn cho bảng")
    ap.add_argument("--no-merge-rows", action="store_true", help="Không tự nối hàng thiếu cột vào hàng trước")
    # chấp nhận --start/--end cho quen tay (bỏ qua)
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--end",   type=int, default=None)
    args = ap.parse_args()
    p.add_argument(
        "--no-merge-rows",
        dest="no_merge_rows",
        action="store_true",
        help="Không gộp 2 dòng header đầu vào 1 (giữ nguyên từng dòng)."
    )

    in_root = Path(args.in_root).resolve()
    out_root = Path(args.out_root).resolve()
    if not in_root.exists():
        print(f"⚠️ in-root không tồn tại: {in_root}"); return
    ensure_dir(out_root)

    files = walk_text_files(in_root, args.pattern)
    if not files:
        print("⚠️ Không tìm thấy file theo pattern"); return

    print(f"🧭 {len(files)} file — TABLES only\n📂 Input : {in_root}\n📦 Output: {out_root}")
    done = skipped = 0
    for i, p in enumerate(sorted(files), 1):

        status, p_tsv, p_jsonl = process_one_file(
            input_txt=p, in_root=in_root, out_root=out_root,
            skip_existing=args.skip_existing,
            min_cols=args.min_cols, max_cols=args.max_cols,
            target_cols=args.target_cols,
            no_merge_rows=getattr(args, "no_merge_rows", False),
        )

        if status == "skip":
            skipped += 1; print(f"[{i}/{len(files)}] ⏭️  {p.relative_to(in_root)} (đã có)")
        else:
            done += 1;   print(f"[{i}/{len(files)}] ✅ {p.relative_to(in_root)} -> {p_tsv.relative_to(out_root)} ; JSONL -> {p_jsonl.relative_to(out_root)}")
    print(f"🎯 Hoàn tất. Xử lý: {done}, bỏ qua: {skipped}, tổng: {len(files)}")

if __name__ == "__main__":
    main()
