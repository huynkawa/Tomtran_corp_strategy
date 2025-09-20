import os
from langchain_chroma import Chroma
from src.config import make_embeddings

def main():
    vector_dir = os.getenv("VECTOR_DIR", "vector_store")
    vectordb = Chroma(
        persist_directory=vector_dir,
        embedding_function=make_embeddings()
    )
    print(f"✅ Đã load vector store: {vector_dir}")
    print("Nhập câu hỏi (gõ 'exit' để thoát).\n")

    while True:
        query = input("❓ Câu hỏi: ").strip()
        if query.lower() in {"exit", "quit"}:
            print("👋 Thoát.")
            break

        results = vectordb.similarity_search_with_score(query, k=3)
        if not results:
            print("⚠️ Không tìm thấy chunk nào.\n")
        else:
            for i, (doc, score) in enumerate(results, 1):
                print(f"\n--- Chunk {i} | Score={score:.4f} ---")
                print(doc.page_content[:800])
                print("Metadata:", doc.metadata)
            print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()
