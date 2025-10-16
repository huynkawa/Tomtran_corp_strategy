# =============================
# src/schemas/meta.py
# Backward‑compatible schema layer for your pipeline
# =============================
from __future__ import annotations
from typing import List, Optional, Literal, Dict, Any, Union, get_args
from pydantic import BaseModel, Field
from datetime import date, datetime
import os, hashlib

# ---- Types
SchemaVersion = Literal["1.0"]

Domain = Literal[
    "finance_bctc",
    "insurance_doc",
    "insurance_book",
    "strategy_book",
    "management_book",
    "internal_corporate",   # unified name (replaces legacy internal_process_uic)
    "other",
]

SourceType = Literal["pdf", "docx", "xlsx", "txt", "image", "web"]
ContentType = Literal["text", "table", "mixed", "image"]
Rights = Literal["public", "internal", "confidential"]
PeriodType = Literal["as_of", "year_ended", "quarter_ended"]
StatementType = Literal["BS", "PL", "CF", "Notes", "Others"]
DocKind = Literal[
    "UW_Guideline", "Risk_Survey", "Policy", "Endorsement",
    "Reinsurance", "Claims", "Procedure"
]

# ---- Models
class SourceInfo(BaseModel):
    path: str
    name: str
    type: SourceType
    ingested_at: datetime = Field(default_factory=datetime.utcnow)

class OCRInfo(BaseModel):
    engine: str = "none"
    version: Optional[str] = None
    dpi: Optional[int] = None
    quality_score: float = 0.0

class TextStats(BaseModel):
    chars: int = 0
    tokens: int = 0

class PageRange(BaseModel):
    start: int = 1
    end: int = 1

class Period(BaseModel):
    type: Optional[PeriodType] = None
    as_of_date: Optional[date] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    fiscal_year: Optional[int] = None

class Company(BaseModel):
    name: Optional[str] = None
    ticker: Optional[str] = None
    company_id: Optional[str] = None

class BookInfo(BaseModel):
    title: Optional[str] = None
    isbn: Optional[str] = None
    publisher: Optional[str] = None
    year: Optional[int] = None
    edition: Optional[str] = None

class StructureInfo(BaseModel):
    chapter: Optional[str] = None
    section: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None

class MetaBase(BaseModel):
    schema_version: SchemaVersion = "1.0"
    doc_id: str
    domain: Domain
    source: SourceInfo
    language: str = "vi"
    content_type: ContentType = "text"
    rights: Rights = "internal"
    title: Optional[str] = None
    subtitle: Optional[str] = None
    authors: List[str] = []
    tags: List[str] = []
    checksum: Optional[str] = None
    ocr: OCRInfo = OCRInfo()
    text_stats: TextStats = TextStats()
    page_range: PageRange = PageRange()

class MetaBCTC(MetaBase):
    company: Company = Company()
    period: Period = Period()
    statement_type: Optional[StatementType] = None
    currency: Optional[str] = None
    multiplier: Optional[int] = None  # 1, 1000, 1000000
    audited: Optional[bool] = None
    auditor_name: Optional[str] = None

class MetaInsuranceDoc(MetaBase):
    doc_kind: Optional[DocKind] = None
    lob: List[str] = []
    jurisdiction: Optional[str] = None
    effective_date: Optional[date] = None

class MetaBook(MetaBase):
    book: BookInfo = BookInfo()
    structure: StructureInfo = StructureInfo()

# Legacy class kept for backward-compat, but Domain now favors internal_corporate
class MetaInternalProcess(MetaBase):
    department: Optional[str] = None
    process_name: Optional[str] = None
    version: Optional[str] = None
    effective_date: Optional[date] = None

# Preferred class for internal documents & reports
class MetaInternalCorporate(MetaBase):
    internal_kind: Optional[Literal[
        "Process", "Procedure", "SOP", "Policy", "Guideline",
        "ERM_Risk_Report", "Management_Report", "KPI_Report",
        "Budget", "Meeting_Minutes", "Compliance",
        "IT_Security", "HR_Policy", "Training_Material"
    ]] = None
    department: Optional[str] = None
    owner: Optional[str] = None
    approver: Optional[str] = None
    process_name: Optional[str] = None
    policy_id: Optional[str] = None
    version: Optional[str] = None
    effective_date: Optional[date] = None
    expiry_date: Optional[date] = None
    status: Optional[Literal["active", "draft", "archived"]] = None
    retention_until: Optional[date] = None

# ---- Simple helpers used by builders
_DEF_SOURCE_TYPE_BY_EXT = {
    ".pdf": "pdf", ".docx": "docx", ".xlsx": "xlsx", ".xls": "xlsx",
    ".txt": "txt", ".png": "image", ".jpg": "image", ".jpeg": "image"
}

def _guess_source_type(filename: str) -> SourceType:
    return _DEF_SOURCE_TYPE_BY_EXT.get(os.path.splitext(filename)[1].lower(), "txt")  # type: ignore

# Prefer the new internal_corporate, but keep old mapping for legacy paths
_DEF_DOMAIN_HINTS = [
    ("finance_bctc", ["bctc", "financial", "balance", "cashflow", "kqkd", "bcđkt"]),
    ("insurance_doc", ["uw", "underwriting", "risk", "policy", "endorsement", "reinsurance", "claim"]),
    ("strategy_book", ["strategy", "chiendluoc", "chienluoc"]),
    ("management_book", ["marketing", "accounting", "hr", "human resource", "quantrinhansu"]),
    ("internal_corporate", [
        "quytrinh", "procedure", "process", "uic", "policy", "chinh sach",
        "erm", "quan tri rui ro", "bao cao quan tri", "minutes", "nghi quyet",
        "budget", "kpi", "okr", "training", "dao tao"
    ]),
    ("internal_process_uic", ["internal_process_uic"])  # last-resort legacy token
]

def _infer_domain_from_path(path: str, default: Domain = "other") -> Domain:
    p = (path or "").lower()
    for dom, keys in _DEF_DOMAIN_HINTS:
        if any(k in p for k in keys):
            return dom  # type: ignore
    return default

# ---- ID/hash helpers (kept minimal so we don’t import utils here)

def _content_sha1(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8", errors="ignore")).hexdigest()

# ---- Builders (new & legacy)

def build_meta(config: Union[Dict[str, Any], None] = None, **kwargs) -> Dict[str, Any]:
    """
    Flexible builder. Accepts either a single config dict or **kwargs (legacy shim).
    Returns a plain dict so downstream json.dump(...) is simple.
    Prefer using utils.meta_utils.build_meta for Pydantic model returns.
    """
    cfg = dict(config or {})
    cfg.update(kwargs or {})

    input_path = str(cfg.get("input_path") or cfg.get("path") or "")
    base_name = os.path.basename(input_path) or cfg.get("base_slug") or "doc"
    domain_hint = cfg.get("class") or cfg.get("domain_hint")
    allowed = set(get_args(Domain))
    domain: Domain = domain_hint if (isinstance(domain_hint, str) and domain_hint in allowed) else _infer_domain_from_path(input_path)
    source_type: SourceType = _guess_source_type(base_name)

    # stable-ish id using path + sha1 prefix (randomized tail should be added in utils layer)
    doc_id = cfg.get("doc_id") or f"doc_{_content_sha1(input_path)[:12]}"

    meta: Dict[str, Any] = {
        "schema_version": "1.0",
        "doc_id": doc_id,
        "domain": domain,
        "source": {
            "path": input_path,
            "name": base_name,
            "type": source_type,
            "ingested_at": datetime.utcnow().isoformat() + "Z",
        },
        "language": cfg.get("language", "vi"),
        "content_type": cfg.get("content_type", "text"),
        "rights": cfg.get("rights", "internal"),
        "title": cfg.get("title"),
        "subtitle": cfg.get("subtitle"),
        "authors": cfg.get("authors") or [],
        "tags": cfg.get("tags") or [],
        "checksum": _content_sha1(cfg.get("clean_text", "")),
        "ocr": {"engine": (cfg.get("ocr_engine") or "none"), "version": cfg.get("ocr_version"), "dpi": cfg.get("ocr_dpi"), "quality_score": cfg.get("ocr_quality", 0.0)},
        "text_stats": {"chars": len(cfg.get("clean_text" , "")), "tokens": 0},
        "page_range": {"start": int(cfg.get("page_start", 1)), "end": int(cfg.get("page_end", cfg.get("page_start", 1)))},
        # light pipeline hints (optional)
        "_pipeline": {"name": cfg.get("pipeline"), "stage": cfg.get("pipeline_stage"), "has_tables": cfg.get("has_tables"), "table_count": cfg.get("table_count")},
    }

    # keep door open for domain specific overrides
    extra = cfg.get("extra") or {}
    for k, v in extra.items():
        if v is None:
            continue
        meta[k] = v

    return meta

# Legacy wrapper expected by older scripts: new_meta(...)

def new_meta(*args, **kwargs) -> Dict[str, Any]:
    """Compatibility shim: maps various call styles to build_meta()."""
    if args and isinstance(args[0], dict):
        return build_meta(args[0])
    return build_meta(**kwargs)

# Provide a legacy MetaDoc symbol (alias to the modern base model) for type hints
class MetaDoc(MetaBase):
    pass

__all__ = [
    "Domain",
    "MetaBase", "MetaBCTC", "MetaInsuranceDoc", "MetaBook", "MetaInternalProcess", "MetaInternalCorporate",
    "MetaDoc",
    "build_meta", "new_meta",
]
