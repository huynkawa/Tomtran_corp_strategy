from langchain_chroma import Chroma
from src.config import make_embeddings
import os

VECTOR_DIR = os.getenv("VECTOR_DIR", "vector_store")
SUBDIR = r"cleaned_scan_data/uic_data"   # chỉnh đúng tên thư mục con bạn đã build

def test_queries(queries, k=2):
    vectordb = Chroma(
        persist_directory=os.path.join(VECTOR_DIR, SUBDIR),
        embedding_function=make_embeddings()
    )

    try:
        count = vectordb._collection.count()
        print(f"📊 Tổng số document trong vector store: {count}")
    except Exception as e:
        print("⚠️ Không đếm được số doc:", e)

    for q in queries:
        print(f"\n🔎 Query: {q}")
        results = vectordb.similarity_search(q, k=k)
        if not results:
            print("❌ Không tìm thấy kết quả.")
        else:
            for i, r in enumerate(results, 1):
                print(f"--- Kết quả {i} ---")
                print("Nguồn:", r.metadata.get("source"))
                print("Nội dung:", r.page_content[:300])

if __name__ == "__main__":
    queries = [
        "doanh thu của UIC",
        "nợ phải trả",
        "vốn chủ sở hữu",
        "nợ dài hạn",
        "tài sản cố định",
        "dòng tiền cuối năm 2024"
    ]
    test_queries(queries, k=2)
