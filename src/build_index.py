import os
import sys
from typing import List
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.ingest import load_documents
import src.env
from tqdm import tqdm

DATA_DIR   = os.getenv("DATA_DIR", "data")
INPUT_DIR  = os.getenv("INPUT_DIR", "inputs")
VECTOR_DIR = os.getenv("VECTOR_DIR", "vector_store")
EMB_MODEL  = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

os.makedirs(VECTOR_DIR, exist_ok=True)

def chunk_documents_with_progress(docs: List[Document], chunk_size=900, chunk_overlap=120) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_chunks = []
    print("[build_index] Chunking tài liệu...")
    for doc in tqdm(docs, desc="Chunking", unit="file"):
        print(f"[build_index] ➕ Đang index file mới: {doc.metadata.get('source', 'unknown')}")
        chunks = splitter.split_documents([doc])
        all_chunks.extend(chunks)
        if len(doc.page_content.strip()) == 0:
            pct = 0
        else:
            pct = min(100, int(len(" ".join(c.page_content for c in chunks)) / len(doc.page_content) * 100))
        print(f" -> {doc.metadata.get('source', 'unknown')} chunked ~{pct}%")
    return all_chunks

def build_index(force_rebuild: bool = False, incremental: bool = True):
    embeddings = HuggingFaceEmbeddings(model_name=EMB_MODEL)

    # 🚀 Nếu rebuild toàn bộ
    if force_rebuild or not (os.path.exists(VECTOR_DIR) and len(os.listdir(VECTOR_DIR)) > 0):
        print("[build_index] 🚀 Bắt đầu build index mới...")
        docs: List[Document] = []
        for p in (DATA_DIR, INPUT_DIR):
            docs += load_documents(p)

        # gắn thêm mtime
        for doc in docs:
            src = doc.metadata.get("source")
            if src and os.path.exists(src):
                doc.metadata["mtime"] = os.path.getmtime(src)

        chunks = chunk_documents_with_progress(
            docs,
            chunk_size=int(os.getenv("CHUNK_SIZE", "900")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "120"))
        )

        vectordb = Chroma.from_documents(
            chunks,
            embedding=embeddings,
            persist_directory=VECTOR_DIR
        )
        vectordb.persist()
        print(f"[build_index] ✅ Build index mới. chunks={len(chunks)}")
        return vectordb

    # 🔄 Nếu load lại index cũ
    print("[build_index] 🔄 Đã load index cũ.")
    vectordb = Chroma(persist_directory=VECTOR_DIR, embedding_function=embeddings)

    if incremental:
        print("[build_index] 🔍 Kiểm tra thay đổi (thêm / xóa file)...")

        # 📂 Lấy danh sách file hiện có trên ổ cứng
        current_files = set()
        for root, _, files in os.walk(INPUT_DIR):
            for f in files:
                current_files.add(os.path.abspath(os.path.join(root, f)))
        for root, _, files in os.walk(DATA_DIR):
            for f in files:
                current_files.add(os.path.abspath(os.path.join(root, f)))

        # 📂 Lấy danh sách file đang có trong DB
        existing_sources = set()
        ids_to_delete = []
        all_data = vectordb.get()
        for idx, meta in enumerate(all_data["metadatas"]):
            if "source" in meta:
                existing_sources.add(os.path.abspath(meta["source"]))
                # nếu file đã bị xóa trên ổ cứng → thêm vào danh sách xóa
                if os.path.abspath(meta["source"]) not in current_files:
                    ids_to_delete.append(all_data["ids"][idx])

        # ❌ Xóa vector của file đã xóa trên ổ cứng
        if ids_to_delete:
            vectordb.delete(ids=ids_to_delete)
            vectordb.persist()
            print(f"[build_index] ❌ Đã xóa {len(ids_to_delete)} vectors của file không còn tồn tại.")

        # ➕ Thêm file mới
        docs: List[Document] = []
        for p in (DATA_DIR, INPUT_DIR):
            docs += load_documents(p)

        # gắn thêm mtime
        for doc in docs:
            src = doc.metadata.get("source")
            if src and os.path.exists(src):
                doc.metadata["mtime"] = os.path.getmtime(src)

        new_docs = [doc for doc in docs if os.path.abspath(doc.metadata.get("source")) not in existing_sources]

        if new_docs:
            chunks = chunk_documents_with_progress(new_docs)
            vectordb.add_documents(chunks)
            vectordb.persist()
            print(f"[build_index] ✅ Đã thêm {len(chunks)} chunks từ {len(new_docs)} file mới.")
        else:
            print("[build_index] ✅ Không có file mới để thêm.")

    return vectordb

if __name__ == "__main__":
    force = "--rebuild" in sys.argv
    db = build_index(force_rebuild=force, incremental=True)
