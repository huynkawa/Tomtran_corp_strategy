# 📁 src/build_index_clean_ready.py
# Build vector store từ dữ liệu sạch trong inputs/clean_ready

import os, json, shutil, argparse, hashlib
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from src.config import make_embeddings
from src.build_index_clean_ocr_data import (
    chunk_documents, load_clean_docs, dedupe_docs_by_content
)

load_dotenv()

INPUT_DIR = Path(r"D:/1.TLAT/3. ChatBot_project/1_Insurance_Strategy - Copy/inputs/clean_ready")
VECTOR_BASE = Path(r"D:/1.TLAT/3. ChatBot_project/1_Insurance_Strategy - Copy/vector_store/clean_ready")

def get_relative_path(base: Path, target: Path) -> Path:
    """Trả về đường dẫn tương đối của target so với base."""
    try:
        return target.relative_to(base)
    except ValueError:
        return target

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table-mode", choices=["rows", "sheet"], default="rows")
    ap.add_argument("--chunk-size", type=int, default=900)
    ap.add_argument("--chunk-overlap", type=int, default=120)
    ap.add_argument("--dedupe", choices=["on", "off"], default="on")
    ap.add_argument("--index-overview", choices=["on", "off"], default="off")
    args = ap.parse_args()

    built, skipped = 0, 0

    all_files = list(INPUT_DIR.rglob("*"))
    valid_ext = (".xlsx", ".csv", ".txt")

    for f in all_files:
        if not f.is_file() or f.suffix.lower() not in valid_ext:
            continue

        rel_path = get_relative_path(INPUT_DIR, f)
        vec_dir = VECTOR_BASE / rel_path.parent / f.stem

        company_tag = f.stem.strip().lower()
        os.makedirs(vec_dir.parent, exist_ok=True)

        choice = "y"
        if os.path.exists(vec_dir):
            choice = input(f"⚠️ Vector store '{vec_dir}' đã tồn tại.\n"
                           "  y = xoá build lại | a = append thêm | n = bỏ qua  → ").strip().lower()

        if choice == "y":
            shutil.rmtree(vec_dir, ignore_errors=True)
            print(f"🗑️ Đã xoá {vec_dir}")
        elif choice == "n":
            print(f"⏭️ Bỏ qua {f.name}")
            skipped += 1
            continue
        elif choice == "a":
            print(f"➕ Append dữ liệu vào {vec_dir}")
        else:
            print("❌ Lựa chọn không hợp lệ, bỏ qua.")
            skipped += 1
            continue

        print(f"\n📥 Nạp dữ liệu từ file: {f}")
        docs = load_clean_docs(str(f), table_mode=args.table_mode,
                               index_overview=(args.index_overview == "on"),
                               company_tag=company_tag)
        if args.dedupe == "on":
            docs = dedupe_docs_by_content(docs)

        chunks = chunk_documents(docs, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
        os.makedirs(vec_dir, exist_ok=True)
        embeddings = make_embeddings()

        if choice == "a" and os.path.exists(vec_dir):
            db = Chroma(persist_directory=str(vec_dir), embedding=embeddings)
            db.add_documents(tqdm(chunks, desc=f"[append] {company_tag}", unit="chunk"))
            db.persist()
        else:
            Chroma.from_documents(
                documents=tqdm(chunks, desc=f"[embedding] {company_tag}", unit="chunk"),
                embedding=embeddings,
                persist_directory=str(vec_dir)
            )

        print(f"✅ Đã lưu vector: {vec_dir} | docs={len(docs)} | chunks={len(chunks)}")
        built += 1

    print("\n=== TÓM TẮT BUILD VECTOR ===")
    print(f"📂 Tổng số vector đã build: {built}")
    print(f"⏭️ Tổng số bỏ qua: {skipped}")

if __name__ == "__main__":
    main()
