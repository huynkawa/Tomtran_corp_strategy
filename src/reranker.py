# src/reranker.py
from sentence_transformers import CrossEncoder
from src.config import RERANK_MODEL   # ✅ lấy model từ config.py

print(f"[reranker] 🚀 Using re-ranker model: {RERANK_MODEL}")
reranker = CrossEncoder(RERANK_MODEL)


def rerank(query, docs, top_n=3, return_scores=False):
    """
    Re-rank các documents dựa trên độ liên quan với query.
    - query: câu hỏi gốc
    - docs: list[Document] từ vector store
    - top_n: số lượng doc muốn giữ lại
    - return_scores: nếu True -> trả về [(doc, score)], ngược lại -> chỉ trả về [doc]
    """
    if not docs:
        return [] if return_scores else []
    
    pairs = [(query, d.page_content) for d in docs]
    scores = reranker.predict(pairs)

    scored = list(zip(docs, scores))
    scored = sorted(scored, key=lambda x: x[1], reverse=True)

    if return_scores:
        return scored[:top_n]
    return [d for d, _ in scored[:top_n]]
