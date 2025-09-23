# 📁 src/build_index_scan_data.py
# === Dùng để xử lý dữ liệu đã qua OCR & đã clean (Excel, CSV, TXT) từ scan ===

import os
import shutil
from pathlib import Path
from typing import List

import pandas as pd
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader, DirectoryLoader, Docx2txtLoader, TextLoader
)
from langchain_chroma import Chroma
from dotenv import load_dotenv
from src.config import make_embeddings

# === Cấu hình ===
load_dotenv()
FINAL_DIR = "inputs/cleaned_scan_input"  # đầu ra của clean_ocr.py
VECTOR_DIR = os.getenv("VECTOR_DIR", "vector_store")

# === Tạo mô tả bảng từ DataFrame ===
def table_to_text(df: pd.DataFrame, source: str, table_name: str = "") -> str:
    lines = []
    columns = [str(c).strip() for c in df.columns]
    if table_name:
        lines.append(f"Bảng: {table_name}")
    lines.append("Tiêu đề cột: " + " | ".join(columns))
    for idx, row in df.iterrows():
        row_str = ", ".join(f"{columns[i]} = {str(val).strip()}" for i, val in enumerate(row))
        lines.append(f"Dòng {idx+1}: {row_str}")
    return "\n".join(lines)

# === Chunk văn bản ===
def chunk_documents(docs: List[Document], chunk_size=900, chunk_overlap=120) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_chunks = []
    for doc in docs:
        if doc.metadata.get("type") == "table":
            all_chunks.append(doc)
        else:
            all_chunks.extend(splitter.split_documents([doc]))
    return all_chunks

# === Load file clean từ OCR (Excel, CSV, TXT) ===
def load_clean_docs(dir_path: str) -> List[Document]:
    docs = []
    for file in Path(dir_path).rglob("*"):
        ext = file.suffix.lower()
        try:
            if ext == ".xlsx":
                xl = pd.ExcelFile(file)
                for sheet in xl.sheet_names:
                    df = xl.parse(sheet)
                    docs.append(Document(
                        page_content=table_to_text(df, str(file), f"Sheet {sheet}"),
                        metadata={"source": str(file), "sheet": sheet, "type": "table"}
                    ))
            elif ext == ".csv":
                df = pd.read_csv(file)
                docs.append(Document(
                    page_content=table_to_text(df, str(file), f"CSV {file.stem}"),
                    metadata={"source": str(file), "type": "table"}
                ))
            elif ext == ".txt":
                docs.extend(TextLoader(str(file), encoding="utf-8").load())
        except Exception as e:
            print(f"❌ Lỗi đọc file {file}: {e}")
    return docs

# === Main ===
from pathlib import Path

if __name__ == "__main__":
    if os.path.exists(VECTOR_DIR):
        shutil.rmtree(VECTOR_DIR)
        print(f"🗑️ Đã xoá vector cũ: {VECTOR_DIR}")

    final_root = Path(FINAL_DIR)
    for subdir in final_root.iterdir():
        if subdir.is_dir():
            print(f"\n📥 Đang nạp dữ liệu từ: {subdir}")

            docs = load_clean_docs(str(subdir))   # hoặc load_documents nếu file bạn có
            chunks = chunk_documents(docs)

            sub_vector_dir = os.path.join(VECTOR_DIR, subdir.name)
            os.makedirs(sub_vector_dir, exist_ok=True)

            vectordb = Chroma.from_documents(
                documents=chunks,
                embedding=make_embeddings(),
                persist_directory=sub_vector_dir
            )

            print("\n=== TÓM TẮT ===")
            print(f"📂 Thư mục: {subdir.name}")
            print(f"📄 Tổng document: {len(docs)}")
            print(f"🧩 Tổng chunks: {len(chunks)}")
            print(f"📊 Bảng (giữ nguyên): {len([d for d in chunks if d.metadata.get('type') == 'table'])}")
            print(f"✅ Vector lưu tại: {sub_vector_dir}")
