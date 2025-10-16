# -*- coding: utf-8 -*-
"""
utils/meta_builder.py — Lắp ráp meta chuẩn cho mọi pipeline (TXT+META).
Kết hợp chặt với utils/ids_meta.py và utils/quality_metrics.py.
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .ids_meta import (
    base_slug_from_path,
    make_doc_id,
    make_doc_id_from_path,  # nếu bạn thêm theo mục A
    sha1_text,
    sha1_path,
)

ISO = "%Y-%m-%dT%H:%M:%SZ"

def utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime(ISO)

def make_meta_for_doc(
    source_path: str | Path,
    page_start: Optional[int] = None,
    page_end: Optional[int] = None,
    base_slug: Optional[str] = None,
    pipeline_name: str = "",
    pipeline_version: str = "",
    pipeline_params: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Tạo khung meta thống nhất để các script điền thêm và/hoặc cập nhật.
    - source_path: file gốc (PDF/DOCX/IMG/…)
    - page_start/end: cửa sổ trang (nếu áp dụng)
    - base_slug: nếu None sẽ lấy từ tên file
    - pipeline_*: ghi dấu pipeline sinh ra file này
    - extra: mọi trường tùy chỉnh (company/period/statement/etc.)
    """
    p = Path(source_path)
    base = base_slug or base_slug_from_path(p)
    doc_id = make_doc_id(base, page_start, page_end)  # hoặc make_doc_id_from_path(...)

    meta: Dict[str, Any] = {
        "source": {
            "path": str(p),
            "name": p.name,
            "stem": p.stem,
            "ext": p.suffix.lower(),
            "abs_path": str(p.resolve()),
        },
        "doc": {
            "base_slug": base,
            "doc_id": doc_id,
            "page_start": page_start,
            "page_end": page_end,
        },
        "hash": {
            "source_path_sha1": sha1_path(p),
            "txt_sha1": None,   # sẽ điền sau khi có TXT
        },
        "pipeline": {
            "name": pipeline_name,
            "version": pipeline_version,
            "params": pipeline_params or {},
        },
        "quality": {},           # sẽ được fill bởi quality_metrics
        "created_utc": utcnow_iso(),
        "updated_utc": None,
    }

    if extra:
        # Gắn thêm các thông tin nghiệp vụ (company, unit, period, statement, language, ...)
        meta.setdefault("biz", {}).update(extra)

    return meta

def stamp_txt_hash(meta: Dict[str, Any], txt_str: Optional[str] = None, txt_path: Optional[str | Path] = None) -> None:
    """
    Cập nhật hash nội dung TXT để pipeline sau biết có thay đổi hay không.
    Chỉ cần truyền 1 trong 2: txt_str hoặc txt_path.
    """
    if txt_str is None and txt_path is None:
        return
    if txt_str is None and txt_path is not None:
        txt_str = Path(txt_path).read_text(encoding="utf-8", errors="ignore")
    meta.setdefault("hash", {})
    meta["hash"]["txt_sha1"] = sha1_text(txt_str or "")

def touch_updated(meta: Dict[str, Any]) -> None:
    meta["updated_utc"] = utcnow_iso()
