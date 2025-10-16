# =============================
# src/utils/meta_utils.py
# Utilities: build_meta (doc-level) & to_vector_meta (chunk-level, flattened)
# =============================
from __future__ import annotations
import os, re
from typing import Any, Dict, Optional, Union

# --- tolerate both schema variants (internal_corporate vs internal_process_uic)
# --- robust imports: chạy được cả `python -m src.xxx` lẫn `python src/xxx.py`
try:
    from src.schemas.meta import (
        MetaBase, MetaBCTC, MetaInsuranceDoc, MetaBook,
        SourceInfo, OCRInfo, TextStats, PageRange, Company, Period,
        BookInfo, StructureInfo,   # <<== THÊM 2 LỚP NÀY
    )
except ModuleNotFoundError:
    from schemas.meta import (
        MetaBase, MetaBCTC, MetaInsuranceDoc, MetaBook,
        SourceInfo, OCRInfo, TextStats, PageRange, Company, Period,
        BookInfo, StructureInfo,   # <<== THÊM 2 LỚP NÀY
    )


# internal domain model (ưu tiên tên mới; fallback tên legacy)
try:
    from src.schemas.meta import MetaInternalCorporate as _MetaInternal
    _INTERNAL_DOMAIN = "internal_corporate"
except Exception:
    try:
        from schemas.meta import MetaInternalCorporate as _MetaInternal
        _INTERNAL_DOMAIN = "internal_corporate"
    except Exception:
        try:
            from src.schemas.meta import MetaInternalProcess as _MetaInternal
        except ModuleNotFoundError:
            from schemas.meta import MetaInternalProcess as _MetaInternal
        _INTERNAL_DOMAIN = "internal_process_uic"

try:
    from src.utils.ids_meta import make_doc_id, content_sha1
except ModuleNotFoundError:
    from utils.ids_meta import make_doc_id, content_sha1

# --- Inference helpers ---------------------------------------------------
_DEF_SRC_EXT = {
    ".pdf":"pdf", ".docx":"docx", ".xlsx":"xlsx", ".xls":"xlsx",
    ".txt":"txt", ".png":"image", ".jpg":"image", ".jpeg":"image"
}

def _guess_source_type(name: str) -> str:
    return _DEF_SRC_EXT.get(os.path.splitext(name)[1].lower(), "txt")


def infer_domain(path: str, hint: Optional[str] = None) -> str:
    """Very light path-based inference; hint wins if provided."""
    if hint:
        return hint
    p = (path or "").lower()
    if any(k in p for k in ["bctc", "financial", "balance", "kqkd", "cashflow", "bcđkt"]):
        return "finance_bctc"
    if any(k in p for k in ["uw", "underwriting", "risk", "policy", "endorsement", "reinsurance", "claim"]):
        return "insurance_doc"
    if any(k in p for k in ["insurance_book", "ins_book"]):
        return "insurance_book"
    if any(k in p for k in ["strategy", "chienluoc", "chiendluoc"]):
        return "strategy_book"
    if any(k in p for k in ["marketing", "accounting", "hr", "human resource", "quantrinhansu"]):
        return "management_book"
    if any(k in p for k in [
        "quytrinh","procedure","process","uic","policy","chinh sach",
        "erm","quan tri rui ro","bao cao quan tri","minutes","nghi quyet",
        "budget","kpi","okr","training","dao tao"
    ]):
        return _INTERNAL_DOMAIN
    return "other"

# --- Heuristics for BCTC fields -----------------------------------------
RE_CURRENCY  = re.compile(r"\b(VND|USD|EUR|JPY)\b", re.I)
RE_FY        = re.compile(r"(năm\s+\d{4}|fiscal\s*year\s*\d{4})", re.I)
RE_MULT      = re.compile(r"(nghìn|thousand|triệu|million|tỷ|billion)", re.I)
RE_COMPANY = re.compile(
    r"(công ty|tập đoàn|tổng công ty)\s+([A-Za-zÀ-ỹ0-9\-\&\,\. ]{3,})",
    re.I
)


def _bctc_hints(text: str) -> Dict[str, Any]:
    t = text or ""
    mcur = RE_CURRENCY.search(t); currency = mcur.group(1).upper() if mcur else None
    mf   = RE_FY.search(t); fy = None
    if mf:
        import re as _re
        y = _re.findall(r"(\d{4})", mf.group(0)); fy = int(y[0]) if y else None
    mm  = RE_MULT.search(t); multiplier = None
    if mm:
        w = mm.group(1).lower()
        multiplier = 1000 if w in ("nghìn","thousand") else (1_000_000 if w in ("triệu","million") else None)
    mco = RE_COMPANY.search(t); company_name = mco.group(2).strip() if mco else None
    return {"currency": currency, "fiscal_year": fy, "multiplier": multiplier, "company_name": company_name}

# --- Builders ------------------------------------------------------------

def build_meta(
    source_path: str,
    *,
    domain_hint: Optional[str] = None,
    text: str = "",
    language: str = "vi",
    content_type: str = "text",
    title: Optional[str] = None,
    ocr_info: Optional[OCRInfo] = None,
    page_start: int = 1,
    page_end: int = 1,
    extra: Optional[Dict[str, Any]] = None,
) -> MetaBase:
    """
    Create a doc-level metadata object (Pydantic model) matching your schema.
    Compatible with both internal_corporate and internal_process_uic variants.
    """
    # --- (1) Ép kiểu & chuẩn hoá an toàn ---
    source_path = str(source_path or "")
    source_name = os.path.basename(source_path) or "doc"
    eff_page_start = int(page_start or 1)
    eff_page_end = int(page_end if page_end is not None else eff_page_start)
    if eff_page_end < eff_page_start:
        eff_page_end = eff_page_start

    doc_id  = make_doc_id(source_path)
    domain  = infer_domain(source_path, domain_hint)
    source  = SourceInfo(path=source_path, name=source_name, type=_guess_source_type(source_name))
    checksum = content_sha1(text or "")

    # (tuỳ chọn) tự động set mixed khi tài liệu có bảng
    if extra and extra.get("has_tables") and content_type == "text":
        # comment dòng dưới nếu bạn muốn luôn giữ "text"
        # content_type = "mixed"
        pass

    base_kwargs = dict(
        doc_id=doc_id,
        domain=domain,
        source=source,
        language=language or "vi",
        content_type=content_type,   # "text" | "table" | "mixed" | "image"
        rights="internal",
        title=title,
        checksum=checksum,
        page_range=PageRange(start=eff_page_start, end=eff_page_end),
        ocr=ocr_info or OCRInfo(engine="none"),
        text_stats=TextStats(chars=len(text or ""), tokens=0),
    )

    # --- (2) Khởi tạo model theo domain ---
    if domain == "finance_bctc":
        hints = _bctc_hints(text)
        meta = MetaBCTC(**base_kwargs)
        meta.company = Company(name=hints.get("company_name"))
        meta.period  = Period(fiscal_year=hints.get("fiscal_year"))
        meta.currency   = hints.get("currency")
        meta.multiplier = hints.get("multiplier")

    elif domain in ("insurance_doc", "insurance_book", "strategy_book", "management_book"):
        meta = MetaBook(**base_kwargs) if domain.endswith("_book") else MetaInsuranceDoc(**base_kwargs)

    elif domain == _INTERNAL_DOMAIN:
        meta = _MetaInternal(**base_kwargs)

    else:
        meta = MetaBase(**base_kwargs)

    # --- (3) Áp override top-level (như cũ) ---
    if extra:
        for k, v in extra.items():
            if hasattr(meta, k) and v is not None:
                setattr(meta, k, v)

    # --- (4) Áp override NESTED theo domain (mới) ---
    if extra:
        # BCTC
        if isinstance(meta, MetaBCTC):
            # company / period
            company_name = extra.get("company_name") or extra.get("company")
            if company_name:
                meta.company = meta.company or Company()
                meta.company.name = company_name
            fy = extra.get("fiscal_year")
            if fy is not None:
                meta.period = meta.period or Period()
                try:
                    meta.period.fiscal_year = int(fy)
                except Exception:
                    pass
            if "statement_type" in extra and extra["statement_type"]:
                meta.statement_type = extra["statement_type"]
            if "currency" in extra and extra["currency"]:
                meta.currency = extra["currency"]
            if "multiplier" in extra and extra["multiplier"] is not None:
                try:
                    meta.multiplier = int(extra["multiplier"])
                except Exception:
                    pass

        # Insurance doc
        if isinstance(meta, MetaInsuranceDoc):
            if "doc_kind" in extra and extra["doc_kind"]:
                meta.doc_kind = extra["doc_kind"]
            if "lob" in extra and extra["lob"]:
                if isinstance(extra["lob"], str):
                    meta.lob = [x.strip() for x in extra["lob"].split(",") if x.strip()]
                elif isinstance(extra["lob"], list):
                    meta.lob = [str(x).strip() for x in extra["lob"] if str(x).strip()]
            if "jurisdiction" in extra and extra["jurisdiction"]:
                meta.jurisdiction = extra["jurisdiction"]
            if "effective_date" in extra and extra["effective_date"]:
                meta.effective_date = extra["effective_date"]  # để nguyên string/ISO; Pydantic sẽ parse nếu hợp lệ

        # Book (insurance/strategy/management_book)
        if isinstance(meta, MetaBook):
            ex = extra or {}  # an toàn nếu extra=None

            # --- title (ưu tiên: YAML -> title param -> meta.title -> file_stem) ---
            file_stem = os.path.splitext(os.path.basename(source_path))[0]
            book_title = ex.get("book_title") or title or meta.title or file_stem
            meta.book = meta.book or BookInfo()
            if not meta.book.title:
                meta.book.title = str(book_title)

            # --- authors (nhận list hoặc chuỗi "a, b, c") ---
            authors = ex.get("authors")
            if isinstance(authors, str):
                authors = [a.strip() for a in authors.split(",") if a.strip()]
            if isinstance(authors, list) and authors:
                meta.authors = [str(a).strip() for a in authors if str(a).strip()]

            # --- các trường book tuỳ chọn ---
            for fld in ("isbn", "publisher", "year", "edition"):
                val = ex.get(fld)
                if val is not None:
                    setattr(meta.book, fld, val)

            # --- structure: chapter/section + page_start/page_end ---
            meta.structure = meta.structure or StructureInfo()
            if "chapter" in ex: meta.structure.chapter = ex["chapter"]
            if "section" in ex: meta.structure.section = ex["section"]

            # mặc định lấy theo page_range nếu YAML không chỉ định
            _ps = ex.get("page_start", page_start or 1)
            _pe = ex.get("page_end", (page_end if page_end is not None else (page_start or 1)))
            try:
                meta.structure.page_start = int(_ps) if _ps is not None else meta.structure.page_start
            except Exception:
                pass
            try:
                meta.structure.page_end = int(_pe) if _pe is not None else meta.structure.page_end
            except Exception:
                pass


        # Internal (corporate/process)
        if isinstance(meta, _MetaInternal):
            for fld in ("internal_kind", "department", "owner", "approver",
                        "process_name", "policy_id", "version",
                        "effective_date", "expiry_date", "status", "retention_until"):
                if fld in extra and extra[fld] is not None and hasattr(meta, fld):
                    setattr(meta, fld, extra[fld])

    return meta


# ===== Helpers: merge JSON-safe + lọc company cho sách =====
import json as _json, re, os
from datetime import date, datetime
from typing import Dict, Any

def _model_to_json_dict(obj) -> Dict[str, Any]:
    """Pydantic v2/v1 -> dict JSON-safe (datetime -> iso)."""
    try:
        return obj.model_dump(mode="json")   # v2
    except Exception:
        try:
            return _json.loads(obj.json())   # v1
        except Exception:
            return getattr(obj, "dict", lambda: dict(obj))()

def _looks_like_book_by_name(path_or_stem: str) -> bool:
    s = os.path.splitext(os.path.basename(str(path_or_stem)))[0].lower()
    return any(k in s for k in [
        "hbr guide", "guide to", "playing to win", "lafley", "porter", "chapter", "isbn", "textbook"
    ])

def _text_mentions_uic(text: str) -> bool:
    patt = r"\b(uic|u\.i\.c|bảo\s*hiểm\s*uic|united\s+insurance|uic\s*vietnam)\b"
    return bool(re.search(patt, (text or "").lower()))

def finalize_meta(meta_from_yaml: Dict[str, Any],
                  meta_core_obj,
                  *,
                  text: str,
                  file_path) -> Dict[str, Any]:
    """
    - JSON-safe hóa meta_core (datetime -> iso)
    - Nếu domain là *_book hoặc tên file trông như sách:
        + XÓA 'class' do YAML đẩy nhầm
        + XÓA 'company' trừ khi text có 'UIC'
    - Trả về dict cuối cùng để ghi ra _meta.json
    """
    base = dict(meta_from_yaml or {})
    core = meta_core_obj if isinstance(meta_core_obj, dict) else _model_to_json_dict(meta_core_obj)

    is_book = str(core.get("domain", "")).endswith("_book") or _looks_like_book_by_name(file_path)
    if is_book:
        base.pop("class", None)
        if "company" in base and not _text_mentions_uic(text):
            base.pop("company", None)

    # merge: giá trị mới (core) ghi đè YAML sau khi lọc
    return {**base, **core}





# Accepts both Pydantic model or plain dict

def _get(obj, path: str, default=None):
    cur = obj
    for part in path.split("."):
        if cur is None: return default
        if isinstance(cur, dict):
            cur = cur.get(part, default)
        else:
            cur = getattr(cur, part, default)
    return cur if cur not in (None, "", []) else default

# --- Auto hints for domain + book extras (works for all books) ----------
from typing import Tuple, List

def auto_meta_hints(
    file_path,                 # Path | str
    ext: str,                  # ".pdf" | ".docx" ...
    text: str = "",            # combined_text (optional)
    yaml_class: Optional[str] = None,  # meta.get("class") if any
) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Return (domain_hint, extras_for_build_meta).
    - Detects *_book by path/filename and PDF doc-info
    - Extracts book_title/authors/publisher/year if available
    - Leaves everything None if nothing found
    """
    import os, re
    p = str(file_path).lower()
    stem = os.path.splitext(os.path.basename(str(file_path)))[0].lower()

    # 1) start from YAML mapping if present
    domain_map = {
        "financials": "finance_bctc", "bctc": "finance_bctc",
        "policy": "insurance_doc", "insurance": "insurance_doc",
        "strategy": "strategy_book", "management": "management_book", "book": "strategy_book",
    }
    domain_hint = domain_map.get((yaml_class or "").lower()) if yaml_class else None

    # 2) sniff PDF doc-info (optional, safe)
    pdf_info: Dict[str, str] = {}
    if ext == ".pdf":
        try:
            import pdfplumber
            with pdfplumber.open(str(file_path)) as pdf:
                meta = getattr(pdf, "metadata", None) or getattr(pdf, "docinfo", None) or {}
                for k, v in (meta.items() if hasattr(meta, "items") else []):
                    kk = str(k).lstrip("/").lower()
                    if v:
                        pdf_info[kk] = str(v)
        except Exception:
            pass

    # 3) global “book” heuristics (filename or doc-info)
    looks_like_book = any(k in stem for k in [
        "hbr guide", "guide to", "playing to win", "lafley", "porter", "chapter", "isbn", "textbook"
    ]) or ("title" in pdf_info or "author" in pdf_info)

    if looks_like_book and (domain_hint in (None, "insurance_doc")):
        # default to strategy_book (you can refine by folder later)
        domain_hint = "strategy_book"

    # 4) build extras (book_title/authors/publisher/year)
    extras: Dict[str, Any] = {}
    if looks_like_book or (domain_hint and domain_hint.endswith("_book")):
        # title → PDF Title → None (let build_meta fallback to file stem)
        if pdf_info.get("title"):
            extras["book_title"] = pdf_info["title"]
        # authors: allow "A;B" or "A, B"
        if pdf_info.get("author"):
            raw = pdf_info["author"].replace(";", ",")
            authors: List[str] = [a.strip() for a in raw.split(",") if a.strip()]
            if authors:
                extras["authors"] = authors
        # publisher/year from Producer/CreationDate/ModDate
        if pdf_info.get("producer"):
            extras["publisher"] = pdf_info["producer"]
        for key in ("creationdate", "moddate"):
            val = pdf_info.get(key)
            if not val:
                continue
            m = re.search(r"(19|20)\d{2}", val)
            if m:
                try:
                    extras["year"] = int(m.group(0))
                    break
                except Exception:
                    pass

    return domain_hint, extras

def to_vector_meta(meta: Union[MetaBase, Dict[str, Any]],
                   chunk_idx: int,
                   page: Optional[int] = None,
                   heading_path: Optional[str] = None) -> Dict[str, Any]:
    """Flatten into compact keys for vector-store filters."""
    out = {
        "doc_id":      _get(meta, "doc_id"),
        "domain":      _get(meta, "domain"),
        "title":       _get(meta, "title"),
        "lang":        _get(meta, "language"),
        "rights":      _get(meta, "rights"),
        "source_path": _get(meta, "source.path"),
        "content_type":_get(meta, "content_type"),
        "page": page,
        "chunk_index": chunk_idx,
        "heading_path": heading_path,
    }
    # BCTC
    out["company"]        = _get(meta, "company.name")      or out.get("company")
    out["statement_type"] = _get(meta, "statement_type")     or out.get("statement_type")
    out["currency"]       = _get(meta, "currency")           or out.get("currency")
    out["multiplier"]     = _get(meta, "multiplier")         or out.get("multiplier")
    out["fiscal_year"]    = _get(meta, "period.fiscal_year") or out.get("fiscal_year")
    # Insurance
    out["doc_kind"]       = _get(meta, "doc_kind")           or out.get("doc_kind")
    lob_list               = _get(meta, "lob") or []
    if isinstance(lob_list, list) and lob_list:
        out["lob"] = ",".join(lob_list)
    # Book
    out["book_title"]     = _get(meta, "book.title") or _get(meta, "title")
    out["isbn"]           = _get(meta, "book.isbn")
    out["chapter"]        = _get(meta, "structure.chapter")
    out["section"]        = _get(meta, "structure.section")
    out["page_start"]     = _get(meta, "structure.page_start")
    out["page_end"]       = _get(meta, "structure.page_end")
    # Internal
    out["internal_kind"]  = _get(meta, "internal_kind")
    out["department"]     = _get(meta, "department")
    out["version"]        = _get(meta, "version")
    out["effective_date"] = _get(meta, "effective_date")
    out["status"]         = _get(meta, "status")

    # drop Nones/empties
    return {k: v for k, v in out.items() if v not in (None, "", [])}


