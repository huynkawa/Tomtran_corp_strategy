# -*- coding: utf-8 -*-
"""
src/a1_text_only_runner.py — Text-first extractor (PDF/DOCX/TXT/IMG/EXCEL)
- Ưu tiên đọc TEXT layer thật sạch; vẫn nhận diện TABLE cơ bản (giữ logic hiện tại)
- KHÔNG sinh vector.jsonl ở bước này (để post-clean rồi mới vectorize)

Điểm chính:
- Hỏi 1 lần Y/N/A cho *toàn bộ lượt chạy*:
    Y = xoá file cũ rồi in lại
    N = chỉ tạo text + meta mới nếu chưa có (bỏ qua nếu đã tồn tại)
    A = append thêm vào file cũ (nếu phát hiện nội dung mới)
- Hỏi 1 lần Y/N bật GPT cho TABLE; TEXT không dùng GPT
- Xuất: <name>_text.txt + <name>_meta.json (không tạo *_vector.jsonl)

Cách chạy:
    python -m src.a1_text_only_runner --start 1 --end 3
"""

import os, re, io, json, hashlib, argparse, shutil, datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import pdfplumber
from PIL import Image, ImageOps
import pytesseract
from docx import Document
from tqdm import tqdm
from langdetect import detect
import yaml
import pandas as pd

# ===== ENV & GPT enhancer (bảng) =====
try:
    import src.env  # nạp OPENAI_API_KEY nếu có
except Exception:
    pass

try:
    from src.gpt_enhancer import enhance_with_gpt as _enhance_with_gpt
except Exception:
    def _enhance_with_gpt(text, meta=None, image=None, **kwargs):
        return text  # fallback an toàn

# ===== Cấu hình mặc định =====
OCR_LANG_DEFAULT = "vie+eng"
OCR_PSM_DEFAULT  = "6"  # single block
OCR_CFG_TEMPLATE = "--psm {psm} preserve_interword_spaces=1"

# Đường dẫn của bạn (giữ nguyên)
INPUT_DIR  = r"D:\1.TLAT\3. ChatBot_project\1_Insurance_Strategy\inputs\b_mix_text_table_inputs_test"
OUTPUT_DIR = r"D:\1.TLAT\3. ChatBot_project\1_Insurance_Strategy\outputs\b1_mix_text_table_output"
YAML_TEXT_PATH_DEFAULT  = r"D:\1.TLAT\3. ChatBot_project\1_Insurance_Strategy\configs\b1_text_only.yaml"
YAML_TABLE_PATH_DEFAULT = r"D:\1.TLAT\3. ChatBot_project\1_Insurance_Strategy\configs\b1_mix_text_table.yaml"

# ===== Print config only once =====
_CONFIG_PRINTED = False
def _print_config_once(args):
    global _CONFIG_PRINTED
    if _CONFIG_PRINTED: return
    _CONFIG_PRINTED = True
    print("=== CẤU HÌNH (TEXT-ONLY runner) ===")
    print(f"📂 INPUT_DIR     : {INPUT_DIR}")
    print(f"📦 OUTPUT_DIR    : {OUTPUT_DIR}")
    print(f"📝 YAML_TEXT     : {YAML_TEXT_PATH_DEFAULT}")
    print(f"📐 YAML_TABLE    : {YAML_TABLE_PATH_DEFAULT}")
    print(f"🔤 OCR_LANG      : {args.ocr_lang} | PSM={args.ocr_psm}")
    print(f"📄 PAGES (PDF)   : {args.start} → {args.end or 'END'} | file_index={args.file_index or 'ALL PDFs'}")
    print("🧾 VECTOR.JSONL  : OFF (post-clean rồi mới vectorize)")
    print("===============================")

# ===== Tiện ích chung =====
def sha1_of_text(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()

def detect_language_safe(text: str) -> str:
    try:
        return detect(text or "")
    except Exception:
        return "unknown"

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_yaml_rules(path: Optional[str]) -> dict:
    if not path: return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}

def deep_merge(a: dict, b: dict) -> dict:
    """Gộp dict b vào a (đệ quy). List thì nối; dict merge; kiểu khác -> lấy b."""
    if not isinstance(a, dict) or not isinstance(b, dict):
        return b
    out = dict(a)
    for k, v in b.items():
        if k in out:
            if isinstance(out[k], dict) and isinstance(v, dict):
                out[k] = deep_merge(out[k], v)
            elif isinstance(out[k], list) and isinstance(v, list):
                out[k] = out[k] + v
            else:
                out[k] = v
        else:
            out[k] = v
    return out

def _merge_files_priority_text(files_text, files_table):
    """Ghép list 'files' của YAML TEXT (ưu tiên) + YAML TABLE, loại trùng theo thứ tự."""
    files_text  = files_text  or []
    files_table = files_table or []
    seen, combo = set(), []
    for item in (files_text + files_table):
        key = json.dumps(item, ensure_ascii=False, sort_keys=True) if isinstance(item, dict) else str(item)
        if key not in seen:
            seen.add(key); combo.append(item)
    return combo

def match_yaml_meta(file_name: str, rules: dict) -> dict:
    if not rules: return {}
    defaults = rules.get("defaults", {}) or {}
    for r in rules.get("files", []) or []:
        patt = r.get("match")
        if patt and re.search(patt, file_name, flags=re.I):
            one = r.copy(); one.pop("match", None)
            return {**defaults, **one}
    return defaults

# ===== GPT Prompt từ YAML (TABLE ONLY) =====
def build_gpt_prompt_from_yaml(yaml_rules: dict, mode: str) -> str:
    gp = (yaml_rules or {}).get("gpt_prompt") or {}
    policy = (yaml_rules or {}).get("policy") or {}
    parts = []
    if gp.get("common"): parts.append(str(gp["common"]).strip())
    if mode and gp.get(mode): parts.append(str(gp[mode]).strip())
    if policy.get("no_hallucination", True):
        parts.append("• Không bịa/suy diễn nội dung hay số liệu.")
    if policy.get("keep_units", True):
        parts.append("• Không tự đổi đơn vị; giữ nguyên đơn vị và giá trị.")
    if policy.get("no_translation", True):
        parts.append("• Không dịch thuật ngữ chuyên ngành; giữ nguyên ngôn ngữ gốc.")
    if policy.get("allow_reformat_only", True):
        parts.append("• Chỉ tái định dạng/làm sạch/chuẩn hoá; KHÔNG diễn giải.")
    if policy.get("forbid_summary", True):
        parts.append("• Không tóm tắt/nhận xét/diễn giải thêm.")
    if mode == "table_only":
        parts.append("• ĐẦU RA PHẢI LÀ TSV (tab \\t), mỗi hàng một dòng, không mô tả.")
    return "\n".join([p for p in parts if p]).strip()

# ===== Heuristics TEXT/TABLE =====
def is_tableish_line(line: str) -> bool:
    # thận trọng hơn để giảm false-positive bảng
    if "\t" in line:
        return True
    if line.count("|") >= 2:
        return True
    multi_spaces = len(re.findall(r"\s{2,}", line)) >= 3
    if multi_spaces:
        digits = sum(ch.isdigit() for ch in line)
        return digits / max(1, len(line)) > 0.2  # ưu tiên dòng có nhiều số
    return False

def split_text_into_text_vs_table_blocks(text: str) -> List[Dict[str, str]]:
    blocks, buf, buf_is_table = [], [], None
    for raw in (text or "").splitlines():
        line = raw.rstrip()
        line_is_table = is_tableish_line(line)
        if buf_is_table is None:
            buf_is_table = line_is_table; buf = [line]
        elif line_is_table == buf_is_table:
            buf.append(line)
        else:
            blocks.append({"type": "table" if buf_is_table else "paragraph",
                           "content": "\n".join(buf).strip()})
            buf = [line]; buf_is_table = line_is_table
    if buf:
        blocks.append({"type": "table" if buf_is_table else "paragraph",
                       "content": "\n".join(buf).strip()})
    return blocks

def _sanitize_cell(cell) -> str:
    s = "" if cell is None else str(cell)
    return s.replace("\r", " ").replace("\n", " ").strip()

def normalize_to_tsv(rows_2d: List[List[str]]) -> str:
    out = []
    for r in rows_2d:
        out.append("\t".join([_sanitize_cell(c) for c in r]))
    return "\n".join(out)

# ==== PATCH: helpers for better tables/text ====
def _approx_cols(tsv: str) -> int:
    """Ước lượng số cột của 1 block TSV."""
    lines = [ln for ln in (tsv or "").splitlines() if ln.strip()]
    if not lines: return 0
    return max(ln.count("\t") + 1 for ln in lines)

def merge_table_fragments(blocks: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Gộp các 'bảng vụn' 1–2 cột vào bảng liền trước (cùng trang).
    Tránh trộn vào các bảng chuẩn (>=3 cột) như Bảng 1.1 & 1.2.
    """
    out = []
    for b in blocks:
        if b.get("type") == "table" and out and out[-1].get("type") == "table":
            same_page = b.get("page") and (b.get("page") == out[-1].get("page"))
            if same_page:
                c_prev = _approx_cols(out[-1].get("content", ""))
                c_cur  = _approx_cols(b.get("content", ""))
                if c_prev >= 3 and c_cur >= 3:
                    out.append(b)  # 2 bảng chuẩn: để nguyên
                else:
                    # gộp mảnh (ít cột) vào bảng trước
                    out[-1]["content"] = (out[-1].get("content","") + "\n" + b.get("content","")).strip()
                    continue
            else:
                out.append(b)
        else:
            out.append(b)
    return out

def _collapse_spaced_caps(s: str) -> str:
    """
    Ghép chuỗi CHỮ HOA bị tách rời bởi khoảng trắng (OCR noise), ví dụ:
    'G N Ộ Đ C Á T Ự S' -> 'GNỘĐCÁTỰS' (giảm rác trong ma trận/legend).
    """
    return re.sub(r'((?:[A-ZÀ-ỸĐ]\s+){3,}[A-ZÀ-ỸĐ])',
                  lambda m: m.group(0).replace(" ", ""),
                  s)

def _fix_common_vn_typos_in_tsv(s: str) -> str:
    # Chuẩn hoá một số lỗi OCR/đặt font
    s = re.sub(r"xảy\s*xa\b", "xảy ra", s, flags=re.I)
    s = s.replace("", "★")  # hợp nhất ký hiệu sao
    return s

_NOISY_ROW_RE = re.compile(
    r"^(GNỘĐCÁTỰS|KHẢ NĂNG XẢY RA\b|\d+\s*-\s*Không\s*$)", re.I
)


# ===== DOCX =====
def read_docx_paragraphs_and_tables(file_path: Path) -> List[Dict[str, str]]:
    """
    Đọc DOCX theo đúng thứ tự xuất hiện (paragraph↔table), gom văn bản trong cell,
    khử dòng trống, dedup ô do merge, và xuất bảng dưới dạng TSV.
    """
    from docx import Document
    from docx.table import _Cell, Table
    from docx.text.paragraph import Paragraph
    from docx.oxml.text.paragraph import CT_P
    from docx.oxml.table import CT_Tbl

    doc = Document(file_path)

    # -- helpers --------------------------------------------------------------
    def iter_block_items(parent):
        """
        Yield Paragraph và Table theo đúng thứ tự trong parent (document body hoặc cell).
        """
        if isinstance(parent, _Cell):
            parent_elm = parent._tc
        else:
            parent_elm = parent.element.body
        for child in parent_elm.iterchildren():
            if isinstance(child, CT_P):
                yield Paragraph(child, parent)
            elif isinstance(child, CT_Tbl):
                yield Table(child, parent)

    def para_is_list(p: Paragraph) -> bool:
        try:
            return p._p.pPr.numPr is not None
        except Exception:
            return False

    def para_level(p: Paragraph) -> int:
        try:
            ilvl = p._p.pPr.numPr.ilvl.val
            return int(ilvl)
        except Exception:
            return 0

    def para_to_text(p: Paragraph) -> str:
        txt = (p.text or "").strip()
        if not txt:
            return ""
        # tiền tố bullet/number nếu là danh sách
        if para_is_list(p):
            indent = "  " * max(0, para_level(p))
            txt = f"{indent}- {txt}"
        return txt

    def cell_text(cell: _Cell) -> str:
        parts = []
        for pr in cell.paragraphs:
            t = para_to_text(pr)
            if t:
                parts.append(t)
        # join theo khoảng trắng để tránh xuống dòng lặt vặt trong cell
        return " ".join(parts).strip()

    def dedup_consecutive(vals: List[str]) -> List[str]:
        out = []
        for v in vals:
            if v is None: v = ""
            v = str(v).strip()
            if not out or v != out[-1]:
                out.append(v)
        return out

    # -- main ----------------------------------------------------------------
    results: List[Dict[str, str]] = []

    for block in iter_block_items(doc):
        if isinstance(block, Paragraph):
            text = para_to_text(block)
            if text:
                results.append({"type": "paragraph", "content": text})

        elif isinstance(block, Table):
            rows_2d: List[List[str]] = []
            for r in block.rows:
                cells = [cell_text(c) for c in r.cells]
                # dedup các ô trùng do merge dọc/ngang
                cells = dedup_consecutive(cells)
                # bỏ hẳn dòng nếu rỗng toàn bộ
                if any(c.strip() for c in cells):
                    rows_2d.append(cells)

            # cân bằng cột (để TSV đều cột hơn chút)
            if rows_2d:
                max_cols = max(len(r) for r in rows_2d)
                padded = [r + [""] * (max_cols - len(r)) for r in rows_2d]
                tsv = normalize_to_tsv(padded)
                results.append({"type": "table", "content": tsv})

    return results


# ===== PDF =====
def preprocess_pil_for_ocr(img: Image.Image) -> Image.Image:
    gray = img.convert("L")
    return ImageOps.autocontrast(gray)

def read_pdf_text_and_images(
    file_path: Path,
    page_start: int = 1,
    page_end: Optional[int] = None,
    ocr_lang: str = OCR_LANG_DEFAULT,
    ocr_psm: str = OCR_PSM_DEFAULT
) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    if page_start < 1: page_start = 1
    try:
        with pdfplumber.open(file_path) as pdf:
            total = len(pdf.pages)
            if not page_end or page_end > total:
                page_end = total
            for idx in range(page_start - 1, page_end):
                page = pdf.pages[idx]
                text = page.extract_text() or ""
                if text.strip():
                    for b in split_text_into_text_vs_table_blocks(text):
                        if b["content"]:
                            results.append({"type": b["type"], "content": b["content"], "page": idx + 1})

                # bắt bảng trực tiếp từ text layer bằng extract_tables()
                try:
                    tables = page.extract_tables()
                    for tbl in (tables or []):
                        rows = [[_sanitize_cell(c) for c in row] for row in (tbl or [])]
                        if rows:
                            results.append({
                                "type": "table",
                                "content": normalize_to_tsv(rows),
                                "page": idx + 1,
                                "source": "plumber_tables"
                            })
                except Exception:
                    pass

                # ảnh nhúng → OCR → TABLE
                for im in page.images or []:
                    x0, top, x1, bottom = im["x0"], im["top"], im["x1"], im["bottom"]
                    try:
                        pil = page.crop((x0, top, x1, bottom)).to_image(resolution=300).original
                        pil = preprocess_pil_for_ocr(pil)
                        cfg = f"--psm {ocr_psm} preserve_interword_spaces=1"
                        txt = pytesseract.image_to_string(pil, lang=ocr_lang, config=cfg)
                        if txt.strip():
                            results.append({"type": "table", "content": txt.strip(), "page": idx + 1, "source": "image_ocr"})
                    except Exception as e:
                        print(f"⚠️ OCR ảnh lỗi (page {idx+1}): {e}")
    except Exception as e:
        print(f"⚠️ PDF đọc lỗi {file_path.name}: {e}")
    return results

# ===== IMG / TXT / Excel =====
def ocr_image_to_text(file_path: Path, ocr_lang: str, ocr_psm: str) -> str:
    pil = preprocess_pil_for_ocr(Image.open(file_path))
    cfg = f"--psm {ocr_psm} preserve_interword_spaces=1"
    return pytesseract.image_to_string(pil, lang=ocr_lang, config=cfg)

def read_txt(file_path: Path) -> str:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

_XLRD_WARNED = False
def read_excel_as_tsv_blocks(file_path: Path) -> List[Dict[str, str]]:
    global _XLRD_WARNED
    results = []
    try:
        suffix = file_path.suffix.lower()
        engine = "openpyxl" if suffix == ".xlsx" else None
        sheets = pd.read_excel(file_path, sheet_name=None, engine=engine)
        for sheet, df in sheets.items():
            df = df.astype(object).where(pd.notna(df), "")
            rows = [list(map(lambda x: str(x).strip(), df.columns.tolist()))]
            for _, r in df.iterrows():
                rows.append([str(c).strip() for c in r.tolist()])
            results.append({"sheet": str(sheet), "content": normalize_to_tsv(rows)})
    except Exception as e:
        msg = str(e)
        if "xlrd" in msg.lower() or "xls Excel support" in msg:
            if not _XLRD_WARNED:
                print(f"⚠️ Excel đọc lỗi {file_path.name}: thiếu xlrd cho .xls. Cài: pip install xlrd>=2.0.1 (hoặc chuyển .xlsx).")
                _XLRD_WARNED = True
        else:
            print(f"⚠️ Excel đọc lỗi {file_path.name}: {e}")
    return results

# ===== Cleanup TEXT & TABLE (nhẹ + nâng cấp) =====
def sanitize_text_block(text: str, yaml_rules: dict) -> Tuple[str, List[str]]:
    warnings = []
    cfg = (yaml_rules or {}).get("text_cleanup", {}) or {}
    out = text or ""

    # strip characters
    for ch in cfg.get("strip_characters", ["\u00A0", "\t", "\r"]):
        out = out.replace(ch, " ")

    # bỏ '|' lẻ
    out = re.sub(r" ?\| ?", " ", out)

    # normalize spaces
    if cfg.get("normalize_spaces", True):
        out = re.sub(r"[ \t]+", " ", out)

    # drop lines matching remove_patterns
    rem_patts = cfg.get("remove_patterns", [])
    if rem_patts:
        kept = []
        for line in out.splitlines():
            drop = False
            for patt in rem_patts:
                try:
                    if re.search(patt, line):
                        drop = True; break
                except re.error:
                    continue
            if not drop:
                kept.append(line)
        out = "\n".join(kept)

    # Bỏ số trang dòng đơn: "   5   "
    out = "\n".join([ln for ln in out.splitlines() if not re.match(r"^\s*\d+\s*$", ln)])

    # Gỡ gạch nối cuối dòng: "tái- \n bảo" -> "tái bảo"
    out = re.sub(r"-\s*\n\s*", "", out)

    # Nối dòng ngắn với dòng sau (nếu dòng sau bắt đầu chữ thường/số/()
    lines = out.split("\n")
    merged = []
    for i, ln in enumerate(lines):
        if i + 1 < len(lines):
            nxt = lines[i+1]
            if len(ln.strip()) < 60 and re.match(r"^[a-zà-ỹ0-9(]", nxt.strip(), flags=re.I):
                merged.append(ln.rstrip() + " " + nxt.lstrip())
                lines[i+1] = ""
                continue
        if ln != "":
            merged.append(ln)
    out = "\n".join(merged)

    # Sửa lỗi chính tả phổ biến sau OCR/clean
    out = re.sub(r"Rất\s+khó\s+xảy\s*xa\b", "Rất khó xảy ra", out, flags=re.I)
    out = re.sub(r"\bhướng\s*đẫn\b", "hướng dẫn", out, flags=re.I)
    # collapse double newlines
    if cfg.get("collapse_double_newlines", True):
        out = re.sub(r"\n{3,}", "\n\n", out.strip())

    if out.strip() == "":
        warnings.append("text_block_empty_after_sanitize")
    return out.strip(), warnings

def sanitize_tsv_block(tsv: str, yaml_rules: dict) -> Tuple[str, List[str]]:
    warnings = []
    rules = yaml_rules or {}
    clean_rules = (rules.get("table_clean_rules") or {})
    validators = (rules.get("validators") or {})
    lines = [ln for ln in (tsv or "").splitlines()]
    new_lines = []
    min_cols = validators.get("min_columns", 1)
    max_cols = validators.get("max_columns", 1000)
    patt_list = []
    for patt in clean_rules.get("drop_rows_matching", []) or []:
        try:
            patt_list.append(re.compile(patt))
        except re.error:
            continue
    for raw in lines:
        raw = _collapse_spaced_caps(raw)  # khử "G N Ộ Đ C Á T Ự S" → "GNỘĐCÁTỰS"
        raw = _fix_common_vn_typos_in_tsv(raw)     # [ADD] sửa "xảy xa", ký hiệu sao
        if _NOISY_ROW_RE.match(raw.strip()):       # [ADD] bỏ dòng rác/đứt dòng
            continue
        cells = [c.strip() for c in raw.split("\t")]
        if any(rp.search(raw) for rp in patt_list):
            continue

        if clean_rules.get("drop_if_all_empty", True):
            if all((c == "" for c in cells)): continue
        if clean_rules.get("trim_cells", True):
            cells = [c.strip() for c in cells]
        if len(cells) < min_cols:
            warnings.append(f"row_dropped_min_cols:{len(cells)}<{min_cols}"); continue
        if len(cells) > max_cols:
            warnings.append(f"row_trimmed_max_cols:{len(cells)}>{max_cols}"); cells = cells[:max_cols]
        new_lines.append("\t".join(cells))
    cleaned = "\n".join(new_lines).strip()
    if cleaned == "": warnings.append("table_block_empty_after_sanitize")
    return cleaned, warnings

# ======= GPT (TABLE only) =======
def call_gpt_table_if_enabled(text: str, meta: dict, yaml_rules: dict, gpt_table_enabled: bool) -> str:
    if not gpt_table_enabled:
        return text
    prompt = build_gpt_prompt_from_yaml(yaml_rules, mode="table_only")
    payload_meta = {**(meta or {}), "gpt_mode": "table_only", "gpt_prompt": prompt}
    try:
        return _enhance_with_gpt(text, payload_meta, None, prompt=prompt, mode="table_only")
    except TypeError:
        try:
            return _enhance_with_gpt(text, payload_meta, None)
        except TypeError:
            return _enhance_with_gpt(text)

# ===== Ghi file theo chế độ Y/N/A =====
def write_outputs_per_mode(base_path: Path, combined_text: str, meta: dict, mode_yna: str):
    """
    mode_yna:
      'Y' → xoá cũ ghi mới
      'N' → chỉ ghi nếu chưa có (nếu tồn tại thì BỎ QUA)
      'A' → append nếu khác nội dung
    """
    txt_out  = base_path.with_name(base_path.name + "_text.txt")
    meta_out = base_path.with_name(base_path.name + "_meta.json")

    if mode_yna == "Y":
        # xoá cũ (nếu có) rồi ghi lại
        for p in (txt_out, meta_out):
            try:
                if p.exists(): p.unlink()
            except Exception:
                pass
        with open(txt_out, "w", encoding="utf-8") as f:
            f.write(combined_text)
        with open(meta_out, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        return "rewritten"

    if mode_yna == "N":
        if txt_out.exists() or meta_out.exists():
            return "skipped_exists"
        with open(txt_out, "w", encoding="utf-8") as f:
            f.write(combined_text)
        with open(meta_out, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        return "created"

    # mode_yna == "A" → append nếu khác nội dung
    appended = False
    if txt_out.exists():
        with open(txt_out, "r", encoding="utf-8", errors="ignore") as f:
            old = f.read()
        if sha1_of_text(old) != sha1_of_text(combined_text):
            with open(txt_out, "a", encoding="utf-8") as f:
                f.write("\n\n===== APPEND @ {} =====\n".format(datetime.datetime.now().isoformat(timespec="seconds")))
                f.write("\n" + combined_text)
            appended = True
    else:
        with open(txt_out, "w", encoding="utf-8") as f:
            f.write(combined_text)
        appended = True

    # meta: lưu lịch sử append ngắn gọn
    hist = []
    if meta_out.exists():
        try:
            with open(meta_out, "r", encoding="utf-8") as f:
                oldm = json.load(f)
            hist = oldm.get("_append_history", [])
        except Exception:
            pass
    if appended:
        hist.append({"ts": datetime.datetime.now().isoformat(timespec="seconds"),
                     "text_sha1": sha1_of_text(combined_text)})
    meta2 = {**meta, "_append_history": hist}
    with open(meta_out, "w", encoding="utf-8") as f:
        json.dump(meta2, f, ensure_ascii=False, indent=2)
    return "appended" if appended else "unchanged"

# ===== Xử lý 1 file =====
def process_file(
    file_path: Path,
    yaml_rules: dict,
    out_dir: Path,
    ocr_lang: str,
    ocr_psm: str,
    page_start: int,
    page_end: Optional[int],
    gpt_table_enabled: bool
) -> Tuple[str, Optional[str]]:
    file_name = file_path.stem
    ext = file_path.suffix.lower()

    meta: Dict = {
        "file": file_name,
        "source_path": str(file_path.resolve()),
        "ocr_lang": ocr_lang,
        "ocr_psm": ocr_psm,
        "runner": "text_only",
        "gpt_table_enabled": bool(gpt_table_enabled),
    }
    meta.update(match_yaml_meta(file_name, yaml_rules))

    combined: List[str] = []
    has_table = False
    postclean_warnings: List[str] = []

    try:
        if ext == ".docx":
            blocks = read_docx_paragraphs_and_tables(file_path)
            tbl_idx = 0
            for b in blocks:
                if b["type"] == "table":
                    has_table = True
                    tbl_idx += 1
                    tsv_raw = b["content"]
                    tsv_raw = call_gpt_table_if_enabled(tsv_raw, {"doc_type":"DOCX","table_index":tbl_idx}, yaml_rules, gpt_table_enabled)
                    tsv_clean, warns = sanitize_tsv_block(tsv_raw, yaml_rules)
                    postclean_warnings.extend([f"docx_table{tbl_idx}:{w}" for w in warns])
                    combined += [f"### [DOCX] [TABLE {tbl_idx}]", tsv_clean.strip()]
                else:
                    para = b["content"]
                    txt_clean, warns = sanitize_text_block(para, yaml_rules)
                    postclean_warnings.extend([f"docx_text:{w}" for w in warns])
                    combined += [f"### [DOCX] [TEXT]", txt_clean.strip()]

        elif ext == ".pdf":
            blocks = read_pdf_text_and_images(file_path, page_start, page_end, ocr_lang, ocr_psm)
            # gộp các bảng vụn 1–2 cột (đặc biệt ở ma trận trang 7)
            blocks = merge_table_fragments(blocks)

            page_table_count: Dict[int, int] = {}
            for b in blocks:
                btype = b.get("type")
                page_no = b.get("page", 0)
                if btype == "table":
                    has_table = True
                    page_table_count[page_no] = page_table_count.get(page_no, 0) + 1
                    idx_on_page = page_table_count[page_no]
                    tsv_raw = b.get("content","")
                    tsv_raw = call_gpt_table_if_enabled(tsv_raw, {"doc_type":"PDF","page":page_no,"table_index":idx_on_page, "source":b.get("source","text_layer")}, yaml_rules, gpt_table_enabled)
                    tsv_clean, warns = sanitize_tsv_block(tsv_raw, yaml_rules)
                    postclean_warnings.extend([f"pdf_p{page_no}_t{idx_on_page}:{w}" for w in warns])
                    combined += [f"### [PDF page {page_no}] [TABLE {idx_on_page}]", tsv_clean.strip()]
                else:
                    para = b.get("content","")
                    txt_clean, warns = sanitize_text_block(para, yaml_rules)
                    postclean_warnings.extend([f"pdf_p{page_no}_text:{w}" for w in warns])
                    combined += [f"### [PDF page {page_no}] [TEXT]", txt_clean.strip()]

        elif ext in [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"]:
            txt = ocr_image_to_text(file_path, ocr_lang, ocr_psm).strip()
            if txt:
                has_table = True  # coi như bảng ảnh
                tsv_raw = txt
                tsv_raw = call_gpt_table_if_enabled(tsv_raw, {"doc_type":"IMG"}, yaml_rules, gpt_table_enabled)
                tsv_clean, warns = sanitize_tsv_block(tsv_raw, yaml_rules)
                postclean_warnings.extend([f"img_table:{w}" for w in warns])
                combined += [f"### [IMG] [TABLE 1]", tsv_clean.strip()]

        elif ext in [".txt", ".csv"]:
            raw = read_txt(file_path).strip()
            txt_clean, warns = sanitize_text_block(raw, yaml_rules)
            postclean_warnings.extend([f"txt_text:{w}" for w in warns])
            combined += [f"### [TXT] [TEXT]", txt_clean.strip()]

        elif ext in [".xlsx", ".xls"]:
            excel_blocks = read_excel_as_tsv_blocks(file_path)
            for i, b in enumerate(excel_blocks, 1):
                has_table = True
                tsv_raw = b["content"]
                tsv_raw = call_gpt_table_if_enabled(tsv_raw, {"doc_type":"EXCEL","sheet":b["sheet"], "table_index": i}, yaml_rules, gpt_table_enabled)
                tsv_clean, warns = sanitize_tsv_block(tsv_raw, yaml_rules)
                postclean_warnings.extend([f"excel_{b['sheet']}_t{i}:{w}" for w in warns])
                combined += [f"### [EXCEL] [SHEET={b['sheet']}] [TABLE {i}]", tsv_clean.strip()]

        elif ext == ".doc":
            return "skip", f"⚠️ Không hỗ trợ định dạng .doc — bỏ qua: {file_path.name}"
        else:
            return "skip", f"⚠️ Không hỗ trợ định dạng {ext} — bỏ qua: {file_path.name}"

        combined_text = "\n".join([c for c in combined if c.strip()]).strip()

        lang = detect_language_safe(combined_text)
        meta.update({
            "language": lang,
            "has_table": has_table,
            "text_sha1": sha1_of_text(combined_text),
            "page_range_applied": {"start": page_start, "end": page_end} if page_start or page_end else None
        })
        if postclean_warnings:
            meta["postclean_warnings"] = postclean_warnings

        return combined_text, meta  # trả text + meta (để caller ghi theo mode Y/N/A)

    except Exception as e:
        return "error", f"❌ Lỗi xử lý {file_path.name}: {e}"

# ===== CLI =====
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Text-only runner — OCR + clean (NO vector.jsonl here)")
    p.add_argument("--start", type=int, default=1, help="[pages] TRANG bắt đầu (1-based)")
    p.add_argument("--end",   type=int, default=None, help="[pages] TRANG kết thúc (1-based, inclusive); None=đến hết")
    p.add_argument("--file_index", type=int, default=None, help="[pages] Chỉ chạy 1 PDF (1-based); bỏ qua = tất cả")
    p.add_argument("--ocr-lang", default=OCR_LANG_DEFAULT, help="Ngôn ngữ OCR (vd: vie+eng)")
    p.add_argument("--ocr-psm",  default=OCR_PSM_DEFAULT, help="PSM Tesseract (vd: 4/6/11)")
    return p

def main():
    args = build_argparser().parse_args()
    _print_config_once(args)

    input_dir = Path(INPUT_DIR)
    out_dir   = Path(OUTPUT_DIR)
    ensure_dir(out_dir)  # chỉ tạo nếu chưa có; KHÔNG xoá

    # ===== NẠP 2 YAML =====
    yaml_table = load_yaml_rules(YAML_TABLE_PATH_DEFAULT)
    yaml_text  = load_yaml_rules(YAML_TEXT_PATH_DEFAULT)

    # GỘP: TEXT ghi đè TABLE
    yaml_rules = deep_merge(yaml_table, yaml_text)
    if isinstance(yaml_table, dict) and isinstance(yaml_text, dict):
        if yaml_table.get("files") or yaml_text.get("files"):
            yaml_rules["files"] = _merge_files_priority_text(
                yaml_text.get("files"), yaml_table.get("files")
            )

    # ===== HỎI 1 LẦN: Y/N/A cho toàn bộ lượt chạy =====
    print("\n👉 Chế độ ghi file (Y/N/A) cho TOÀN BỘ lượt chạy:")
    print("   Y = XÓA file cũ rồi in mới;  N = chỉ tạo nếu chưa có;  A = append thêm nếu nội dung mới.")
    while True:
        mode_yna = input("   Chọn [Y/N/A]: ").strip().upper()
        if mode_yna in ("Y", "N", "A"):
            break
        print("   Vui lòng nhập Y, N hoặc A.")

    # ===== HỎI 1 LẦN: GPT cho bảng (Y/N) =====
    while True:
        ans = input("👉 Bật GPT cho TABLE? (Y/N, TEXT luôn KHÔNG dùng GPT): ").strip().upper()
        if ans in ("Y", "N"):
            break
        print("   Vui lòng nhập Y hoặc N.")
    gpt_table_enabled = (ans == "Y")

    # ===== Liệt kê input =====
    files = sorted(list(input_dir.rglob("*.*")))

    # ===== LỌC PDF / NON-PDF =====
    pdfs = [Path(f) for f in files if Path(f).suffix.lower() == ".pdf"]
    if args.file_index and 1 <= args.file_index <= len(pdfs):
        files = [pdfs[args.file_index - 1]]
    elif args.file_index and args.file_index > len(pdfs):
        print(f"⚠️ file_index={args.file_index} > số PDF ({len(pdfs)}). Sẽ chạy toàn bộ input.")

    # ===== Chạy tất cả =====
    for fpath in tqdm(files, desc="Processing by pages (PDF) or full (others)"):
        base_out = out_dir / fpath.stem

        # xử lý
        result, meta_or_err = process_file(
            file_path=Path(fpath),
            yaml_rules=yaml_rules,
            out_dir=out_dir,
            ocr_lang=args.ocr_lang,
            ocr_psm=args.ocr_psm,
            page_start=args.start,
            page_end=args.end,
            gpt_table_enabled=gpt_table_enabled
        )

        # ghi theo chế độ
        if isinstance(result, str) and result in ("skip","error"):
            print(meta_or_err)  # message
            continue

        combined_text: str = result
        meta: dict = meta_or_err if isinstance(meta_or_err, dict) else {}
        status = write_outputs_per_mode(base_out, combined_text, meta, mode_yna)
        print(f"✅ {fpath.name} → {status}")

    print("\n🎯 Hoàn tất. (Chưa tạo vector.jsonl; làm post-clean rồi vectorize ở bước sau.)")

if __name__ == "__main__":
    main()
