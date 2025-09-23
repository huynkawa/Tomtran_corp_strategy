from src.config import make_embeddings
from langchain_chroma import Chroma

db = Chroma(persist_directory="vector_store", embedding_function=make_embeddings())
results = db.similarity_search("nợ ngắn hạn", k=3)

for i, r in enumerate(results, 1):
    print(f"\n--- Document {i} ---")
    print(r.page_content[:500])   # in thử 500 ký tự đầu
    print("Metadata:", r.metadata)
