# =============================
# src/utils/ids_meta.py
# ID + hashing helpers (backward‑compatible)
# =============================
from __future__ import annotations
import os, re, hashlib
from uuid import uuid4
from pathlib import Path

# Stable-ish id: path + seed → sha1 prefix + short random tail

def make_doc_id(source_path: str, seed: str = "") -> str:
    h = hashlib.sha1((str(source_path).replace("\\", "/") + "::" + seed).encode("utf-8")).hexdigest()[:12]
    return f"doc_{h}_{uuid4().hex[:8]}"


def make_page_id(doc_id: str, page: int) -> str:
    return f"{doc_id}:p{int(page):04d}"


def make_chunk_id(doc_id: str, idx: int) -> str:
    return f"{doc_id}:{int(idx):05d}"


def content_sha1(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8", errors="ignore")).hexdigest()


def base_slug_from_path(path: str | os.PathLike) -> str:
    stem = Path(path).stem
    slug = re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_")
    return slug or "doc"
