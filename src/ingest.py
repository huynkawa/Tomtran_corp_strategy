# src/ingest.py
import os
import io
from typing import List
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
import pytesseract
from PIL import Image
import fitz  # PyMuPDF để đọc PDF scan

load_dotenv()
DATA_DIR   = os.getenv("DATA_DIR", "data")
INPUT_DIR  = os.getenv("INPUT_DIR", "inputs")
OUT_DIR    = os.getenv("OUTPUT_DIR", "outputs")

os.makedirs(OUT_DIR, exist_ok=True)

def save_ocr_text(file_path: str, text: str):
    """Lưu text OCR ra file .txt để kiểm chứng sau này"""
    out_dir = os.path.join(OUT_DIR, "ocr_texts")
    os.makedirs(out_dir, exist_ok=True)
    base_name = os.path.basename(file_path)
    txt_path = os.path.join(out_dir, base_name + ".txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[OCR] 💾 Lưu text OCR tại: {txt_path}")

def ocr_pdf(file_path: str) -> str:
    """Thực hiện OCR cho PDF scan (trả về text)."""
    text_content = []
    try:
        pdf_doc = fitz.open(file_path)
        for page_num in range(len(pdf_doc)):
            page = pdf_doc[page_num]
            pix = page.get_pixmap()
            # ✅ Dùng io.BytesIO để mở ảnh từ bộ nhớ
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text_content.append(pytesseract.image_to_string(img, lang="eng+vie"))
    except Exception as e:
        print(f"[OCR PDF lỗi] {file_path} → {e}")
    return "\n".join(text_content)

def load_documents(dir_path: str) -> List[Document]:
    docs: List[Document] = []
    if not os.path.isdir(dir_path):
        return docs

    print(f"[load_documents] Đang load từ: {dir_path}")

    # --- PDF
    for file in Path(dir_path).rglob("*.pdf"):
        print(f"🧾 Loading PDF: {file.name}")
        try:
            pdf_docs = PyPDFLoader(str(file)).load()
            # Nếu tất cả trang rỗng → thử OCR
            if all(len(d.page_content.strip()) == 0 for d in pdf_docs):
                print(f"[build_index] 🔍 Phát hiện PDF scan: {file}, đang chạy OCR...")
                text = ocr_pdf(str(file))
                if text.strip():
                    print(f"[build_index] 📖 OCR thành công: {file} (đọc được {len(text)} ký tự)")
                    save_ocr_text(str(file), text)  # ✅ Lưu text ra file .txt
                    pdf_docs = [Document(page_content=text, metadata={"source": str(file)})]
                else:
                    print(f"[build_index] ⚠️ OCR thất bại: {file} (không nhận diện được chữ)")
            docs.extend(pdf_docs)
        except Exception as e:
            print(f"[LỖI PDF] {file.name} → {e}")

    # --- DOCX
    try:
        docs += DirectoryLoader(dir_path, glob="**/*.docx", loader_cls=Docx2txtLoader).load()
    except Exception as e:
        print(f"[LỖI DOCX] {dir_path} → {e}")

    # --- TXT / MD
    try:
        docs += DirectoryLoader(dir_path, glob="**/*.txt", loader_cls=TextLoader).load()
        docs += DirectoryLoader(dir_path, glob="**/*.md", loader_cls=TextLoader).load()
    except Exception as e:
        print(f"[LỖI TXT/MD] {dir_path} → {e}")

    # --- PNG / JPG / JPEG: OCR
    for file in Path(dir_path).rglob("*"):
        if file.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            print(f"🖼️ OCR Image: {file.name}")
            try:
                text = pytesseract.image_to_string(Image.open(file), lang="eng+vie")
                if text.strip():
                    print(f"[build_index] 📖 OCR thành công: {file} (đọc được {len(text)} ký tự)")
                    save_ocr_text(str(file), text)  # ✅ Lưu text ra file .txt
                    docs.append(Document(page_content=text, metadata={"source": str(file)}))
                else:
                    print(f"[OCR] Không đọc được text từ {file.name}")
            except Exception as e:
                print(f"[LỖI OCR] {file.name} → {e}")

    return docs

def chunk_documents(docs: List[Document], chunk_size=900, chunk_overlap=120) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)

if __name__ == "__main__":
    # Ưu tiên DATA_DIR (chuẩn hoá) → nếu trống thì dùng INPUT_DIR
    sources = []
    for p in (DATA_DIR, INPUT_DIR):
        sources += load_documents(p)
    chunks = chunk_documents(sources)

    with open(os.path.join(OUT_DIR, "ingest_stats.txt"), "w", encoding="utf-8") as f:
        f.write(f"Documents: {len(sources)}\nChunks: {len(chunks)}\n")
    print(f"[ingest] Documents={len(sources)} | Chunks={len(chunks)} (log tại {OUT_DIR}/ingest_stats.txt)")
