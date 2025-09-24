import os, shutil, io, json
from pathlib import Path
from typing import List

import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import camelot
import pandas as pd

from dotenv import load_dotenv
from langchain.schema import Document
from langchain_community.document_loaders import (
    DirectoryLoader, PyPDFLoader, Docx2txtLoader, TextLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.config import make_embeddings
from langchain_chroma import Chroma
from tqdm import tqdm   # ✅ progress bar

# === Load env ===
load_dotenv()
DATA_DIR = os.getenv("DATA_DIR", "data")
INPUT_DIR = os.getenv("INPUT_DIR", "inputs")
OUT_DIR = os.getenv("OUTPUT_DIR", "outputs")
VECTOR_DIR = os.getenv("VECTOR_DIR", "vector_store")
OCR_DIR = os.path.join(OUT_DIR, "ocr_texts")
OCR_PDF_DIR = os.path.join(OUT_DIR, "ocr_pdfs")
CSV_DIR = os.path.join(OUT_DIR, "tables")

os.makedirs(OCR_DIR, exist_ok=True)
os.makedirs(OCR_PDF_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)


# === Hàm OCR PDF (PyMuPDF + pytesseract) ===
def ensure_ocr_pdf(file_path: str) -> str:
    try:
        pdf_docs = PyPDFLoader(file_path).load()

        if all(len(d.page_content.strip()) == 0 for d in pdf_docs):
            print(f"[Fallback OCR] {file_path} → Dùng PyMuPDF + pytesseract")

            text_out = os.path.join(OCR_DIR, f"{Path(file_path).stem}.txt")
            with fitz.open(file_path) as doc, open(text_out, "w", encoding="utf-8") as out:
                for page_num in range(len(doc)):
                    pix = doc[page_num].get_pixmap(dpi=300)
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    text = pytesseract.image_to_string(img, lang="vie+eng")
                    out.write(f"\n--- Page {page_num+1} ---\n{text}\n")
            return file_path

    except Exception as e:
        print(f"[OCR lỗi chung] {file_path} → {e}")

    return file_path


# === Hàm chuyển bảng thành text dễ hiểu ===
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


# === Trích xuất bảng từ PDF ===
def extract_tables(file_path: str) -> List[Document]:
    docs = []
    try:
        tables = camelot.read_pdf(file_path, pages="all", flavor="stream")
        for i, table in enumerate(tables):
            df = table.df
            table_text = table_to_text(df, file_path, f"PDF Table {i+1}")
            if table_text.strip():
                docs.append(Document(
                    page_content=table_text,
                    metadata={"source": file_path, "page": i+1, "type": "table"}
                ))
                csv_out = os.path.join(CSV_DIR, f"{Path(file_path).stem}_table_{i+1}.csv")
                df.to_csv(csv_out, index=False, encoding="utf-8-sig")
                print(f"[CSV Export] {csv_out}")
    except Exception as e:
        print(f"[Camelot lỗi] {file_path} → {e}")
    return docs


# === Chunk tài liệu ===
def chunk_documents(docs: List[Document], chunk_size=900, chunk_overlap=120) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_chunks: List[Document] = []
    print("[ingest] Chunking...")
    for doc in docs:
        if doc.metadata.get("type") == "table":
            all_chunks.append(doc)
        else:
            chunks = splitter.split_documents([doc])
            all_chunks.extend(chunks)
    return all_chunks


# === Load documents tổng hợp ===
def load_documents(dir_path: str) -> List[Document]:
    docs: List[Document] = []
    if not os.path.isdir(dir_path):
        return docs
    for file in Path(dir_path).rglob("*"):
        ext = file.suffix.lower()
        if ext == ".pdf":
            real_file = ensure_ocr_pdf(str(file))
            docs.extend(extract_tables(real_file))
            try:
                docs.extend(PyPDFLoader(real_file).load())
            except:
                pass
        elif ext in [".docx"]:
            docs.extend(DirectoryLoader(dir_path, glob="**/*.docx", loader_cls=Docx2txtLoader).load())
        elif ext in [".xls", ".xlsx", ".csv"]:
            try:
                if ext == ".csv":
                    df = pd.read_csv(file)
                    docs.append(Document(
                        page_content=table_to_text(df, str(file), f"CSV {Path(file).stem}"),
                        metadata={"source": str(file), "type": "table"}
                    ))
                else:
                    xl = pd.ExcelFile(file)
                    for sheet in xl.sheet_names:
                        df = xl.parse(sheet)
                        csv_out = os.path.join(CSV_DIR, f"{Path(file).stem}_{sheet}.csv")
                        df.to_csv(csv_out, index=False, encoding="utf-8-sig")
                        print(f"[Excel Export] {csv_out}")
                        docs.append(Document(
                            page_content=table_to_text(df, str(file), f"Sheet {sheet}"),
                            metadata={"source": str(file), "sheet": sheet, "type": "table"}
                        ))
            except Exception as e:
                print(f"[Excel/CSV lỗi] {file} → {e}")
        elif ext in [".txt", ".md"]:
            docs_txt = TextLoader(str(file), encoding="utf-8").load()
            base = str(file).replace("_text.txt", "")
            meta_file = base + "_meta.json"
            if os.path.exists(meta_file):
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                for d in docs_txt:
                    d.metadata.update(meta)
            docs.extend(docs_txt)
    return docs


# === Main ===
if __name__ == "__main__":
    sub_vector_dir = os.path.join(VECTOR_DIR, "raw_clean_data")

    print("\n======================================")
    print("⚙️  Lựa chọn chế độ build dữ liệu")
    print("   - Gõ 'r' rồi Enter → Xoá dữ liệu cũ và build lại từ đầu")
    print("   - Gõ 'c' rồi Enter → Giữ nguyên dữ liệu cũ (không build)")
    print("   - Nhấn Enter (bỏ trống) → mặc định = Continue (c)")
    print("======================================")

    choice = input("👉 Bạn chọn [R/c]: ").strip().lower()

    if choice == "" or choice == "c":
        print("⏭️ Tiếp tục giữ dữ liệu cũ. Không build lại.")
        exit(0)
    elif choice == "r":
        if os.path.exists(sub_vector_dir):
            shutil.rmtree(sub_vector_dir)
            print(f"🗑️ Đã xoá vector cũ: {sub_vector_dir}")
        if os.path.exists(CSV_DIR):
            shutil.rmtree(CSV_DIR)
            os.makedirs(CSV_DIR, exist_ok=True)
            print(f"🗑️ Đã xoá CSV cũ: {CSV_DIR}")
    else:
        print("❌ Lựa chọn không hợp lệ. Thoát.")
        exit(1)

    print(f"\n📥 Đang nạp dữ liệu từ {DATA_DIR} và {INPUT_DIR}")

    sources = []
    for p in (DATA_DIR, INPUT_DIR):
        sources += load_documents(p)

    chunks = chunk_documents(sources)
    print(f"[ingest] Tổng số chunks: {len(chunks)}")

    os.makedirs(sub_vector_dir, exist_ok=True)

    embeddings = make_embeddings()
    vectordb = Chroma.from_documents(
        documents=tqdm(chunks, desc="[embedding raw_data]", unit="chunk"),
        embedding=embeddings,
        persist_directory=sub_vector_dir
    )

    print("\n=== TÓM TẮT ===")
    print(f"📄 Tổng document: {len(sources)}")
    print(f"🔖 Tổng chunks: {len(chunks)}")
    print(f"📊 Bảng (giữ nguyên): {len([d for d in chunks if d.metadata.get('type')=='table'])}")
    print(f"📂 Vector lưu tại: {sub_vector_dir}")
    print(f"📂 CSV bảng lưu tại: {CSV_DIR}")
