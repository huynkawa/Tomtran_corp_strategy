# === Dùng để xử lý dữ liệu đã qua OCR & đã clean (Excel, CSV, TXT) từ scan ===

import os
import shutil
import json
from pathlib import Path
from typing import List

import pandas as pd
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from dotenv import load_dotenv
from src.config import make_embeddings
from tqdm import tqdm   # ✅ hiển thị progress bar
from termcolor import colored  # ✅ để in highlight

# === Cấu hình ===
load_dotenv()
FINAL_DIR = "inputs/cleaned_scan_input"  # đầu ra của clean_ocr.py
VECTOR_DIR = os.getenv("VECTOR_DIR", "vector_store")

HIGHLIGHT_KEYWORDS = ["doanh thu", "phí bảo hiểm", "bảo hiểm", "doanhthu"]

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

# === In preview có highlight ===
def preview_with_highlight(df: pd.DataFrame, file: str, sheet: str = None):
    try:
        preview_df = df.head(5).copy()
        preview_str = ""
        for col in preview_df.columns:
            col_str = str(col)
            if any(k in col_str.lower() for k in HIGHLIGHT_KEYWORDS):
                col_display = colored(col_str, "red", attrs=["bold"])
            else:
                col_display = col_str
            preview_str += f"{col_display}\t"
        preview_str += "\n"

        for _, row in preview_df.iterrows():
            row_display = []
            for col, val in row.items():
                val_str = str(val)
                if any(k in val_str.lower() for k in HIGHLIGHT_KEYWORDS):
                    row_display.append(colored(val_str, "red", attrs=["bold"]))
                else:
                    row_display.append(val_str)
            preview_str += "\t".join(row_display) + "\n"

        print(f"\n📊 Preview bảng từ {file} (sheet {sheet}):\n{preview_str}\n{'-'*60}")
    except Exception as e:
        print(f"⚠️ Không in preview được cho {file}: {e}")

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

# === Load file clean từ OCR (Excel, CSV, TXT + metadata) ===
def load_clean_docs(dir_path: str) -> List[Document]:
    docs = []
    for file in Path(dir_path).rglob("*"):
        ext = file.suffix.lower()
        try:
            if ext == ".xlsx":
                xl = pd.ExcelFile(file)
                for sheet in xl.sheet_names:
                    df = xl.parse(sheet)
                    preview_with_highlight(df, str(file), sheet)
                    docs.append(Document(
                        page_content=table_to_text(df, str(file), f"Sheet {sheet}"),
                        metadata={"source": str(file), "sheet": sheet, "type": "table"}
                    ))
            elif ext == ".csv":
                df = pd.read_csv(file)
                preview_with_highlight(df, str(file), "CSV")
                docs.append(Document(
                    page_content=table_to_text(df, str(file), f"CSV {file.stem}"),
                    metadata={"source": str(file), "type": "table"}
                ))
            elif ext == ".txt":
                docs_txt = TextLoader(str(file), encoding="utf-8").load()
                base = str(file).replace("_text.txt", "")
                meta_file = base + "_meta.json"
                if os.path.exists(meta_file):
                    with open(meta_file, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    for d in docs_txt:
                        d.metadata.update(meta)
                docs.extend(docs_txt)
        except Exception as e:
            print(f"❌ Lỗi đọc file {file}: {e}")
    return docs

# === Main ===
if __name__ == "__main__":
    final_root = Path(FINAL_DIR)

    for subdir in final_root.iterdir():
        if subdir.is_dir():
            sub_vector_dir = os.path.join(VECTOR_DIR, "cleaned_scan_data", subdir.name)

            # ✅ Nếu đã tồn tại vector cho subdir này → hỏi Yes/No
            if os.path.exists(sub_vector_dir):
                choice = input(
                    f"⚠️ Vector store {sub_vector_dir} đã tồn tại. "
                    f"Bạn có muốn xoá và build lại từ đầu? (y/n): "
                ).strip().lower()
                if choice == "y":
                    shutil.rmtree(sub_vector_dir)
                    print(f"🗑️ Đã xoá vector cũ: {sub_vector_dir}")
                else:
                    print(f"⏭️ Bỏ qua {subdir.name}, giữ dữ liệu cũ.")
                    continue

            print(f"\n📥 Đang nạp dữ liệu từ: {subdir}")

            docs = load_clean_docs(str(subdir))
            chunks = chunk_documents(docs)
            print(f"[ingest] Tổng số chunks: {len(chunks)}")

            os.makedirs(sub_vector_dir, exist_ok=True)

            embeddings = make_embeddings()
            vectordb = Chroma.from_documents(
                documents=tqdm(chunks, desc=f"[embedding] {subdir.name}", unit="chunk"),
                embedding=embeddings,
                persist_directory=sub_vector_dir
            )

            print("\n=== TÓM TẮT ===")
            print(f"📂 Thư mục: {subdir.name}")
            print(f"📄 Tổng document: {len(docs)}")
            print(f"🧩 Tổng chunks: {len(chunks)}")
            print(f"📊 Bảng (giữ nguyên): {len([d for d in chunks if d.metadata.get('type') == 'table'])}")
            print(f"✅ Vector lưu tại: {sub_vector_dir}")
