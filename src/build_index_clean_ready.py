# 📁 src/build_index_clean_ocr_data.py
# Ingest dữ liệu đã clean (clean30) vào vector store cho RAG

import os, re, json, shutil, hashlib, argparse
from pathlib import Path
from typing import List, Iterable, Optional

import pandas as pd
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from dotenv import load_dotenv
from tqdm import tqdm

from src.config import make_embeddings

load_dotenv()
FINAL_DIR  = "inputs/cleaned_scan_input"   # đầu ra từ clean30
VECTOR_DIR = os.getenv("VECTOR_DIR", "vector_store")

NUM_PATTERN = re.compile(r"^\s*[-+]?\s*[\d\s.,]+$")

def is_number_like(x):
    return bool(NUM_PATTERN.match(str(x).strip())) if x is not None else False

def to_number(x):
    if x is None: return None
    s = str(x).strip().replace(" ", "")
    if s.count(",")>0 and s.count(".")==0: s = s.replace(",", ".")
    elif s.count(".")>1 and s.count(",")==0:
        parts = s.split("."); s = "".join(parts[:-1]) + "." + parts[-1]
    try: return float(s) if "." in s else int(s)
    except: return x

def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def chunk_documents(docs: List[Document], chunk_size=900, chunk_overlap=120) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    out=[]
    for d in docs:
        if d.metadata.get("type") in ("table_row","table","table_row_value","table_overview"):
            out.append(d)
        else:
            out.extend(splitter.split_documents([d]))
    return out

def _attach_meta_if_exists(doc: Document, txt_path: str):
    base=(txt_path.replace("_text.txt","").replace("_clean.txt","").replace(".txt",""))
    mf=base+"_meta.json"
    if os.path.exists(mf):
        try:
            with open(mf,"r",encoding="utf-8") as f: meta=json.load(f)
            doc.metadata.update(meta)
        except: pass

# ---------- detect loại tài liệu ----------
def detect_doc_type(path: Path) -> str:
    name = path.stem.lower()
    if any(k in name for k in ["bctc","balance","fs","financial"]):
        return "BCTC"
    if any(k in name for k in ["survey","risk","inspection","giamsat"]):
        return "survey"
    return "other"

# ---------- build row docs ----------
def _row_to_period_docs(row: pd.Series, base_meta: dict, line_item_col: Optional[str], company_tag: str, doc_type: str) -> List[Document]:
    docs=[]
    line_item = str(row.get(line_item_col,"") if line_item_col else row.get("line_item","") or "").strip()
    currency  = row.get("currency","")
    scale     = row.get("scale","")
    tag_name  = (row.get("line_item_norm","") or str(line_item).lower()).strip()
    code      = str(row.get("code","")).strip()

    # cờ phân loại
    level = str(row.get("level","")).lower()
    flags=[]
    if bool(row.get("is_total", False)):    flags.append("total")
    if bool(row.get("is_subtotal", False)): flags.append("subtotal")
    if not flags:                            flags.append("leaf")

    # BỎ group không có số
    has_any_value = any(str(row.get(k,"")).strip()!="" for k in ("end","start") if k in row.index)
    if level=="group" and not has_any_value:
        return docs

    # extra columns
    extra_cols = [c for c in row.index if str(c).lower().startswith("extra")]
    extras_text = [f"{c}={row.get(c,'')}" for c in extra_cols if str(row.get(c,'')).strip()!=""]

    for label in ("end","start"):
        if label not in row.index: continue
        val=row.get(label,None)
        if val is None or (isinstance(val,float) and pd.isna(val)): continue
        txt=str(val).strip()
        if txt=="": continue
        num=to_number(val)
        display=num if isinstance(num,(int,float)) else txt

        unit=[]
        if str(currency).strip(): unit.append(str(currency).strip())
        if str(scale).strip():    unit.append(str(scale).strip())
        unit_s = f" ({', '.join(unit)})" if unit else ""

        text = f"{line_item} — {label}: {display}{unit_s}"
        if extras_text: text += " | " + " ; ".join(extras_text)
        if tag_name: text += f" tag:{tag_name}"
        if code:     text += f" tag:code:{code}"
        text += f" tag:company:{company_tag} tag:type:{doc_type}"
        for f in flags: text += f" tag:{f}"

        meta=dict(base_meta, **{
            "type":"table_row_value",
            "period_label":label,
            "currency":str(currency) if currency is not None else "",
            "scale":str(scale) if scale is not None else "",
            "code":code,
            "line_item":line_item,
            "line_item_norm":tag_name,
            "flags":flags,
            "company":company_tag,
            "doc_type":doc_type,
        })
        for c in extra_cols:
            meta[c] = row.get(c,"")

        docs.append(Document(page_content=text, metadata=meta))
    return docs

# ---------- loaders ----------
def docs_from_xlsx(path: Path, table_mode: str, index_overview: bool, company_tag: str) -> Iterable[Document]:
    doc_type = detect_doc_type(path)
    xl=pd.ExcelFile(path)
    for sheet in xl.sheet_names:
        df=xl.parse(sheet)
        df.columns=[str(c).strip() for c in df.columns]

        if index_overview:
            yield Document(
                page_content="Tiêu đề cột: " + " | ".join([str(c) for c in df.columns]) + f" tag:company:{company_tag} tag:type:{doc_type}",
                metadata={"source":str(path), "sheet":sheet, "type":"table_overview", "company":company_tag,"doc_type":doc_type}
            )

        has_period=("end" in df.columns) or ("start" in df.columns)
        line_item_col="line_item" if "line_item" in df.columns else None

        if has_period:
            for idx,row in df.iterrows():
                base={"source":str(path),"sheet":sheet,"row_index":int(idx),"company":company_tag,"doc_type":doc_type}
                for d in _row_to_period_docs(row, base, line_item_col, company_tag, doc_type): yield d
        else:
            if table_mode=="rows":
                for idx,row in df.iterrows():
                    content=" | ".join([f"{c}={row.get(c,'')}" for c in df.columns if str(row.get(c,'')).strip()!=""])
                    content += f" tag:company:{company_tag} tag:type:{doc_type}"
                    yield Document(page_content=content, metadata={"source":str(path),"sheet":sheet,"type":"table_row","row_index":int(idx),"company":company_tag,"doc_type":doc_type})

def docs_from_csv(path: Path, table_mode: str, index_overview: bool, company_tag: str) -> Iterable[Document]:
    doc_type = detect_doc_type(path)
    df=pd.read_csv(path); df.columns=[str(c).strip() for c in df.columns]
    if index_overview:
        yield Document(page_content="Tiêu đề cột: " + " | ".join([str(c) for c in df.columns]) + f" tag:company:{company_tag} tag:type:{doc_type}",
                       metadata={"source":str(path), "type":"table_overview", "company":company_tag,"doc_type":doc_type})
    has_period=("end" in df.columns) or ("start" in df.columns)
    line_item_col="line_item" if "line_item" in df.columns else None

    if has_period:
        for idx,row in df.iterrows():
            base={"source":str(path),"row_index":int(idx),"company":company_tag,"doc_type":doc_type}
            for d in _row_to_period_docs(row, base, line_item_col, company_tag, doc_type): yield d
    else:
        if table_mode=="rows":
            for idx,row in df.iterrows():
                content=" | ".join([f"{c}={row.get(c,'')}" for c in df.columns if str(row.get(c,'')).strip()!=""])
                content += f" tag:company:{company_tag} tag:type:{doc_type}"
                yield Document(page_content=content, metadata={"source":str(path),"type":"table_row","row_index":int(idx),"company":company_tag,"doc_type":doc_type})

def docs_from_txt(path: Path, company_tag: str) -> Iterable[Document]:
    doc_type = detect_doc_type(path)
    docs=TextLoader(str(path), encoding="utf-8").load()
    for d in docs:
        _attach_meta_if_exists(d, str(path))
        d.page_content += f" tag:company:{company_tag} tag:type:{doc_type}"
        d.metadata["company"] = company_tag
        d.metadata["doc_type"] = doc_type
        yield d

def load_clean_docs(path: str, table_mode: str, index_overview: bool, company_tag: str) -> List[Document]:
    p=Path(path)
    if p.is_file():
        files=[p]
    else:
        files=list(p.rglob("*"))
    docs=[]
    for file in files:
        ext=file.suffix.lower()
        try:
            if ext in (".xlsx",".xls"): docs.extend(docs_from_xlsx(file, table_mode, index_overview, company_tag))
            elif ext==".csv":           docs.extend(docs_from_csv(file, table_mode, index_overview, company_tag))
            elif ext==".txt":           docs.extend(docs_from_txt(file, company_tag))
        except Exception as e:
            print(f"❌ Lỗi đọc file {file}: {e}")
    return docs

def dedupe_docs_by_content(docs: List[Document]) -> List[Document]:
    seen=set(); out=[]
    for d in docs:
        h=content_hash(d.page_content)
        if h in seen: continue
        seen.add(h); out.append(d)
    return out

# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table-mode", choices=["rows", "sheet"], default="rows")
    ap.add_argument("--chunk-size", type=int, default=900)
    ap.add_argument("--chunk-overlap", type=int, default=120)
    ap.add_argument("--dedupe", choices=["on", "off"], default="on")
    ap.add_argument("--index-overview", choices=["on", "off"], default="off")
    args = ap.parse_args()

    root = Path(FINAL_DIR)
    built, skipped = 0, 0

    for sub in root.iterdir():

        # === Case 1: folder bất kỳ ===
        if sub.is_dir():
            # nếu folder chứa nhiều file -> build từng file thành vector riêng
            files = list(sub.glob("*.*"))
            if files:
                for f in files:
                    if f.suffix.lower() not in (".xlsx", ".csv", ".txt"):
                        continue
                    company_tag = f.stem.strip().lower()
                    vec_dir = os.path.join(VECTOR_DIR, "cleaned_scan_data", sub.name, company_tag)

                    choice = "y"
                    if os.path.exists(vec_dir):
                        choice = input(f"⚠️ Vector store {vec_dir} đã tồn tại. "
                                       "Chọn y = xoá build lại, a = append thêm dữ liệu, n = bỏ qua: ").strip().lower()

                    if choice == "y":
                        shutil.rmtree(vec_dir, ignore_errors=True)
                        print(f"🗑️ Đã xoá {vec_dir}")
                    elif choice == "n":
                        print(f"⏭️ Bỏ qua {company_tag}")
                        skipped += 1
                        continue
                    elif choice == "a":
                        print(f"➕ Append dữ liệu mới vào {vec_dir}")
                    else:
                        print("❌ Lựa chọn không hợp lệ, bỏ qua.")
                        skipped += 1
                        continue

                    print(f"\n📥 Nạp dữ liệu từ file: {f} (tag={company_tag})")
                    docs = load_clean_docs(str(f), table_mode=args.table_mode,
                                           index_overview=(args.index_overview == "on"),
                                           company_tag=company_tag)
                    if args.dedupe == "on":
                        docs = dedupe_docs_by_content(docs)

                    chunks = chunk_documents(docs, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
                    os.makedirs(vec_dir, exist_ok=True)

                    embeddings = make_embeddings()
                    if choice == "a" and os.path.exists(vec_dir):
                        db = Chroma(persist_directory=vec_dir, embedding=embeddings)
                        db.add_documents(tqdm(chunks, desc=f"[append] {company_tag}", unit="chunk"))
                        db.persist()
                    else:
                        Chroma.from_documents(
                            documents=tqdm(chunks, desc=f"[embedding] {company_tag}", unit="chunk"),
                            embedding=embeddings,
                            persist_directory=vec_dir
                        )
                    print(f"✅ Vector lưu tại: {vec_dir} | docs={len(docs)} | chunks={len(chunks)}")
                    built += 1

            else:
                print(f"⚠️ Folder {sub.name} không có file hợp lệ (.xlsx/.csv/.txt) → bỏ qua.")

        # === Case 2: file trực tiếp trong cleaned_scan_input ===
        elif sub.is_file() and sub.suffix.lower() in (".xlsx", ".csv", ".txt"):
            company_tag = sub.stem.strip().lower()
            vec_dir = os.path.join(VECTOR_DIR, "cleaned_scan_data", company_tag)

            choice = "y"
            if os.path.exists(vec_dir):
                choice = input(f"⚠️ Vector store {vec_dir} đã tồn tại. "
                               "Chọn y = xoá build lại, a = append thêm dữ liệu, n = bỏ qua: ").strip().lower()

            if choice == "y":
                shutil.rmtree(vec_dir, ignore_errors=True)
                print(f"🗑️ Đã xoá {vec_dir}")
            elif choice == "n":
                print(f"⏭️ Bỏ qua {company_tag}")
                skipped += 1
                continue
            elif choice == "a":
                print(f"➕ Append dữ liệu mới vào {vec_dir}")
            else:
                print("❌ Lựa chọn không hợp lệ, bỏ qua.")
                skipped += 1
                continue

            print(f"\n📥 Nạp dữ liệu từ file trực tiếp: {sub} (tag={company_tag})")
            docs = load_clean_docs(str(sub), table_mode=args.table_mode,
                                   index_overview=(args.index_overview == "on"),
                                   company_tag=company_tag)
            if args.dedupe == "on":
                docs = dedupe_docs_by_content(docs)

            chunks = chunk_documents(docs, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
            os.makedirs(vec_dir, exist_ok=True)

            embeddings = make_embeddings()
            if choice == "a" and os.path.exists(vec_dir):
                db = Chroma(persist_directory=vec_dir, embedding=embeddings)
                db.add_documents(tqdm(chunks, desc=f"[append] {company_tag}", unit="chunk"))
                db.persist()
            else:
                Chroma.from_documents(
                    documents=tqdm(chunks, desc=f"[embedding] {company_tag}", unit="chunk"),
                    embedding=embeddings,
                    persist_directory=vec_dir
                )
            print(f"✅ Vector lưu tại: {vec_dir} | docs={len(docs)} | chunks={len(chunks)}")
            built += 1

    # === Tổng kết ===
    print("\n=== TÓM TẮT BUILD VECTOR ===")
    print(f"📂 Tổng số vector đã build: {built}")
    print(f"⏭️ Tổng số bỏ qua: {skipped}")
if __name__ == "__main__":
    main()
