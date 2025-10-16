# -*- coding: utf-8 -*-
"""
src.a2_clean_text_only_prevector.py
— Final CLEAN (TEXT ONLY) trước khi index vào vector store
— Đọc từ outputs\\a1_text_only_outputs → ghi sang outputs\\a2_text_only_prevector (mirror tree)

Thay đổi chính (so với bản trước):
- CHỈ xử lý TEXT (bỏ toàn bộ TABLE/TSV).
- Nâng cấp làm sạch: giữ/chuẩn hoá tiền tệ ($, ₫/đ, VND/VNĐ), chuẩn hoá dấu câu cong → ASCII,
  nối dòng “mềm” không làm dính từ, lọc watermark/ads (Bookey, “SSccaann ttoo DDoowwnnllooaadd”, …).
- Hỏi 1 lần chế độ ghi:
    Y = purge  (xóa 3 file cũ rồi tạo lại mới)
    N = skip   (nếu đã có đủ file đích thì bỏ qua)
    A = append (vector.jsonl: append; text/meta: ghi lại để đồng bộ)
- Lọc theo trang --start/--end dựa trên marker A1: "### [PDF page X] [TEXT]"

Ví dụ:
    python -m src.a2_clean_text_only_prevector --start 26 --end 28
    python -m src.a2_clean_text_only_prevector

Ghi chú:
- Bước này không cần YAML (rule-based). Có thể bổ sung YAML sau (patterns, replace-map) nếu muốn.
"""

import os, re, json, argparse, hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# ============== ĐƯỜNG DẪN MẶC ĐỊNH ==============
IN_ROOT_DEFAULT  = r"D:\1.TLAT\3. ChatBot_project\1_Insurance_Strategy\outputs\a1_text_only_outputs"
OUT_ROOT_DEFAULT = r"D:\1.TLAT\3. ChatBot_project\1_Insurance_Strategy\outputs\a2_text_only_prevector"

# ============== HỎI 1 LẦN CHẾ ĐỘ GHI ==============
def ask_write_mode_once() -> str:
    """
    Y = purge   (xóa file cũ rồi tạo mới)
    N = skip    (nếu đã có đủ file đích -> bỏ qua)
    A = append  (text/meta: ghi lại; vector.jsonl: append)
    """
    print("Chế độ ghi toàn bộ lượt chạy (Y=purge, N=skip, A=append):")
    while True:
        a = input("Chọn [Y/N/A]: ").strip().upper()
        if a in ("Y","N","A"):
            return {"Y":"purge","N":"skip","A":"append"}[a]
        print("Vui lòng nhập Y / N / A.")

# ============== TIỆN ÍCH CƠ BẢN ==============
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def sha1_of_text(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()

def find_pairs(in_root: Path) -> List[Tuple[Path, Path]]:
    """
    Tìm tất cả cặp (text_path, meta_path) theo pattern *_text.txt / *_meta.json
    trong cây thư mục IN_ROOT.
    """
    pairs = []
    for txt_path in in_root.rglob("*_text.txt"):
        meta_path = txt_path.with_name(txt_path.name.replace("_text.txt", "_meta.json"))
        if meta_path.exists():
            pairs.append((txt_path, meta_path))
    return pairs

# ============== CLEAN TEXT (RULE-BASED) ==============
# Dòng rác/header/footer phổ biến
TEXT_BADLINES_PATTERNS = [
    r"^\s*page\s*\d+\s*/\s*\d+\s*$",          # "Page 3/20"
    r"^\s*\d+\s*$",                           # dòng chỉ có số (thường là số trang)
    r"^\s*copyright.*$",
    r"^\s*figure\s*\d+.*$",
    r"^\s*table\s*\d+.*$",
    r"^More\s+Free\s+Books\s+on\s+Bookey.*$", # quảng cáo
    r"SSccaann\s*ttoo\s*DDoowwnnllooaadd",    # watermark méo OCR
    r"^\s*Huy\s*$",                           # rác đơn lẻ (đặt theo phát hiện thực tế)
]

# Map dấu câu "cong" → ASCII/Unicode an toàn cho vector
PUNCT_MAP = {
    "\u2018": "'",   # ‘ → '
    "\u2019": "'",   # ’ → '
    "\u201C": '"',   # “ → "
    "\u201D": '"',   # ” → "
    "\u2013": "-",   # – (en-dash) → hyphen
    "\u2014": " — ", # — (em-dash) → em-dash có khoảng trắng 2 bên
}

# Cho phép (ASCII + tiếng Việt + dấu câu cơ bản + EM DASH '—')
WEIRD_CHARS = r"[^A-Za-z0-9À-ỹ.,:;?!%/\-\(\)\[\]{}_&'\" \t—]"

def _apply_punct_map(text: str) -> str:
    # Ánh xạ dấu câu cong → thẳng
    for k, v in PUNCT_MAP.items():
        text = text.replace(k, v)
    # CHỈ chuẩn hoá khoảng trắng quanh em-dash — (không đụng vào '-')
    text = re.sub(r"\s*—\s*", " — ", text)
    return text

def _normalize_ascii_dash(text: str) -> str:
    """
    Đổi ' - ' thành em-dash ' — ' khi ở giữa 2 token dài (≥2 ký tự).
    Không phá 'A-B', 'Pro-X', v.v.
    """
    # Cách 1 (đơn giản, Unicode \w): trái ≥2 ký tự, phải ≥2 ký tự (dùng lookahead cố định)
    text = re.sub(r'(\b\w{2,})\s-\s(?=\w{2,}\b)', r'\1 — ', text)
    # Cách 2 (bổ sung cho từ có dấu nháy tiếng Việt/Anh như "man's", “năm’s”)
    text = re.sub(r"([A-Za-zÀ-ỹ0-9][A-Za-zÀ-ỹ0-9'’]{1,})\s-\s(?=[A-Za-zÀ-ỹ0-9'’]{2,}\b)", r"\1 — ", text)
    return text


def _soft_join_hyphen(text: str) -> str:
    """
    Nối dòng khi cuối dòng có '-' (hyphen):
    - Vế trái NGẮN (≤3 ký tự)  → GIỮ hyphen: 'no-\nman's' -> 'no-man's'
    - Vế trái DÀI  (≥4 ký tự)  → BỎ hyphen: 'inves-\n tment' -> 'investment'
    (Không dùng look-behind để tránh lỗi variable-width)
    """
    # Giữ hyphen cho tiền tố ngắn
    text = re.sub(r'(\b\w{1,3})-\s*\n(?=\w)', r'\1-', text)
    # Bỏ hyphen cho vế trái dài
    text = re.sub(r'(\b\w{4,})-\s*\n(?=\w)', r'\1', text)
    return text



def _soft_join_sentences(text: str) -> str:
    # nối dòng mềm: nếu giữa hai dòng không kết thúc bằng .!?;: thì nối bằng 1 space
    return re.sub(r"(?<![\.!?;:])\n(?!\n)", " ", text)


# --- Currency normalizer (đặt TRÊN clean_text_block) ---
CURRENCY_PAT_DOLLAR = re.compile(
    r"(?<![A-Za-z])\$(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)"
)
CURRENCY_PAT_VND = re.compile(
    r"\b(?:VNĐ|VND|vnđ|vnd)\s*([0-9][\d\.,]*)"
)
CURRENCY_PAT_DONG = re.compile(
    r"[₫đ]\s*([0-9][\d\.,]*)"
)

def _normalize_currency_text(text: str) -> str:
    text = CURRENCY_PAT_DOLLAR.sub(lambda m: f"USD {m.group(1)}", text)
    text = CURRENCY_PAT_VND.sub(lambda m: f"VND {m.group(1)}", text)
    text = CURRENCY_PAT_DONG.sub(lambda m: f"VND {m.group(1)}", text)
    text = re.sub(r"\b(USD|VND)\s+([0-9])", r"\1 \2", text)
    return text



def _apply_lexical_fixes(t: str) -> str:
    """
    Vá các biến thể OCR của 'no-man's-land':
    - noman's-land / noman’s-land
    - noman’s — land / noman’s - land
    - no man’s land (bị tách khoảng trắng)
    - nomansland (dính liền)
    """
    patterns = [
        r"\bnoman['’]?s?\s*(?:-|—|–)?\s*land\b",  # noman’s-land / noman’s — land
        r"\bno\s*man['’]?\s*(?:-|—|–)?\s*land\b", # no man’s land / no-man’s land
        r"\bnomansland\b",                        # nomansland
    ]
    for pat in patterns:
        t = re.sub(pat, "no-man's-land", t, flags=re.I)
    return t




def clean_text_block(raw: str) -> str:
    t = (raw or "").replace("\u00A0", " ").replace("\t", " ").replace("\r", "")
    t = _apply_punct_map(t)
    t = _normalize_ascii_dash(t)

    t = re.sub(r" ?\| ?", " ", t)            # bỏ '|' lẻ trong text
    t = _normalize_currency_text(t)          # chuẩn hoá tiền tệ trước khi lọc ký tự
    t = _soft_join_hyphen(t)
    t = _soft_join_sentences(t)

    kept = []
    badline_res = [re.compile(p, re.I) for p in TEXT_BADLINES_PATTERNS]
    for line in t.splitlines():
        s = line.strip()
        drop = any(p.search(s) for p in badline_res)
        if not drop:
            kept.append(line)
    t = "\n".join(kept)

    t = re.sub(WEIRD_CHARS, "", t)
    t = re.sub(r"[ \t]{2,}", " ", t)         # chuẩn hoá khoảng trắng
    t = _apply_lexical_fixes(t)

    # chèn khoảng trắng sau dấu ':' nếu trước đó KHÔNG phải là số (tránh 12:30), và sau đó là chữ/số
    t = re.sub(r"(?<!\d):(?=[A-Za-zÀ-ỹ0-9])", ": ", t)

    # collapse 3+ newline → 2 newline
    t = re.sub(r"\n{3,}", "\n\n", t.strip())
    return t.strip()


# ============== PHÂN KHỐI + LỌC THEO TRANG ==============
# A1 marker mẫu: "### [PDF page 26] [TEXT]" hoặc "### [PDF page 26] [TABLE 1]"
MARKER_RE   = re.compile(r"^###\s*\[(?P<src>[^\]]+)\]\s*\[(?P<kind>TEXT|TABLE[^\]]*)\]\s*$", flags=re.I)
PDF_PAGE_RE = re.compile(r"(?i)^PDF\s+page\s+(\d+)$")

def parse_blocks_with_page(mixed_text: str) -> List[Dict]:
    blocks = []
    cur = None
    for line in (mixed_text or "").splitlines():
        m = MARKER_RE.match(line)
        if m:
            # flush block cũ
            if cur and cur.get("buf"):
                cur["content"] = "\n".join(cur["buf"]).strip()
                blocks.append({k: v for k, v in cur.items() if k != "buf"})
            # mở block mới
            src = (m.group("src") or "").strip()
            kind = (m.group("kind") or "").strip().upper()
            page = None
            pm = PDF_PAGE_RE.match(src)
            if pm:
                try:
                    page = int(pm.group(1))
                except Exception:
                    page = None
            cur = {
                "type": ("text" if kind.startswith("TEXT") else "table"),
                "page": page,
                "src": src,
                "kind": kind,
                "buf": []
            }
        else:
            if cur is None:
                # nếu trước đó chưa có marker hợp lệ thì bỏ qua rác header
                continue
            cur["buf"].append(line)
    if cur and cur.get("buf"):
        cur["content"] = "\n".join(cur["buf"]).strip()
        blocks.append({k: v for k, v in cur.items() if k != "buf"})
    return blocks

def filter_blocks_by_page(blocks: List[Dict], start: Optional[int], end: Optional[int]) -> List[Dict]:
    if not start and not end:
        return blocks
    keep = []
    for b in blocks:
        p = b.get("page")
        if p is None:
            # Có filter trang thì chỉ giữ các block PDF có số trang
            continue
        if start and p < start:
            continue
        if end and p > end:
            continue
        keep.append(b)
    return keep

# ============== GHI VECTOR.JSONL ==============
def append_vector_jsonl(path: Path, content: str, meta: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"content": content, "metadata": meta}, ensure_ascii=False) + "\n")

# ============== XỬ LÝ 1 CẶP FILE (TỪ IN_ROOT → OUT_ROOT MIRROR) ==============
def _purge_outputs(out_txt: Path, out_meta: Path, out_vec: Path):
    """Xoá sạch 3 file đích nếu tồn tại (chế độ Y=purge)."""
    for p in (out_txt, out_meta, out_vec):
        try:
            if p.exists():
                p.unlink()
        except Exception:
            pass

def process_pair(
    in_root: Path,
    out_root: Path,
    txt_path: Path,
    meta_path: Path,
    write_mode: str,
    page_start: Optional[int],
    page_end: Optional[int]
):
    # Tính đường dẫn mirror
    rel_dir  = txt_path.parent.relative_to(in_root)        # thư mục con so với IN_ROOT
    base_raw = txt_path.name.replace("_text.txt", "")      # tên gốc (không đuôi)
    out_dir  = out_root / rel_dir
    ensure_dir(out_dir)

    out_txt  = out_dir / f"{base_raw}_text.txt"
    out_meta = out_dir / f"{base_raw}_meta.json"
    out_vec  = out_dir / f"{base_raw}_vector.jsonl"

    # skip nếu đã có
    if write_mode == "skip" and out_txt.exists() and out_meta.exists() and out_vec.exists():
        print(f"⏭️  Skip (đã có đủ 3 file): {out_txt}")
        return

    # purge: xóa cả 3 file trước khi xử lý
    if write_mode == "purge":
        _purge_outputs(out_txt, out_meta, out_vec)

    # === đọc input ===
    raw_text = txt_path.read_text(encoding="utf-8", errors="ignore")
    try:
        meta_in = json.loads(meta_path.read_text(encoding="utf-8", errors="ignore") or "{}")
    except Exception:
        meta_in = {}

    # === tách block theo marker A1 và lọc theo trang (nếu có) ===
    blocks_all = parse_blocks_with_page(raw_text)
    blocks     = filter_blocks_by_page(blocks_all, page_start, page_end)

    cleaned_parts: List[str] = []
    kept_count = 0

    # CHỈ xử lý TEXT; mọi TABLE đều bỏ qua ở bước này
    for b in blocks:
        if b.get("type") != "text":
            continue
        c = (b.get("content") or "").strip()
        if not c:
            continue
        cleaned = clean_text_block(c)
        if cleaned:
            kept_count += 1
            cleaned_parts += [f"### [TEXT CLEAN] [SRC={b.get('src','')}]", cleaned]
            append_vector_jsonl(
                out_vec,
                cleaned,
                {**meta_in, "content_type": "TEXT", "stage": "final_clean",
                 "page": b.get("page"), "src": b.get("src")}
            )

    # Trường hợp không có marker hợp lệ (hiếm) → coi là TEXT nguyên khối
    if not blocks_all:
        cleaned = clean_text_block(raw_text)
        if cleaned:
            kept_count += 1
            cleaned_parts = ["### [TEXT CLEAN] [SRC=UNKNOWN]", cleaned]
            append_vector_jsonl(out_vec, cleaned, {**meta_in, "content_type":"TEXT", "stage":"final_clean"})

    final_text = "\n".join(cleaned_parts).strip()

    # Chuẩn bị META out
    meta_out = dict(meta_in)
    meta_out["final_clean_sha1"] = sha1_of_text(final_text)
    if page_start or page_end:
        meta_out["pages_filtered"] = {"start": page_start, "end": page_end}
    meta_out["clean_blocks_kept"] = kept_count
    meta_out["a2_mode"] = write_mode

    # === Ghi text/meta theo chế độ ===
    # Với 'append': vector đã append ở trên; text/meta ghi lại để đồng bộ nội dung mới.
    if write_mode in ("purge", "append"):
        out_txt.write_text(final_text, encoding="utf-8")
        out_meta.write_text(json.dumps(meta_out, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        # skip-mode: chỉ ghi nếu thiếu
        if not out_txt.exists():
            out_txt.write_text(final_text, encoding="utf-8")
        if not out_meta.exists():
            out_meta.write_text(json.dumps(meta_out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"✅ Wrote: {out_txt.name}, {out_meta.name}, + vector.jsonl ({write_mode}) → {out_dir}")

# ============== CLI ==============
def build_argparser():
    p = argparse.ArgumentParser(description="Clean TEXT trước vectorize (mirror in→out)")
    p.add_argument("--in-root",  default=IN_ROOT_DEFAULT,  help="Thư mục chứa đầu ra A1 (*_text.txt + *_meta.json)")
    p.add_argument("--out-root", default=OUT_ROOT_DEFAULT, help="Thư mục đích A2 (sẽ mirror cây thư mục)")
    p.add_argument("--start", type=int, default=None, help="[PDF] Trang bắt đầu (1-based). Chỉ giữ block 'PDF page X' trong khoảng.")
    p.add_argument("--end",   type=int, default=None, help="[PDF] Trang kết thúc (1-based, inclusive).")
    return p

def main():
    args = build_argparser().parse_args()
    in_root  = Path(args.in_root)
    out_root = Path(args.out_root)
    ensure_dir(out_root)

    write_mode = ask_write_mode_once()   # purge / skip / append
    pairs = find_pairs(in_root)
    if not pairs:
        print("⚠️ Không tìm thấy cặp *_text.txt + *_meta.json nào trong IN_ROOT.")
        return

    total = 0
    for txt_path, meta_path in pairs:
        total += 1
        process_pair(in_root, out_root, txt_path, meta_path, write_mode, args.start, args.end)

    print(f"\n🎯 Hoàn tất A2 (TEXT ONLY): {total} file(s). Output ở: {out_root}")

if __name__ == "__main__":
    main()
