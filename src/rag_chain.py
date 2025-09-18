import os
from typing import Tuple, List
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.llm import call_llm
import src.env

VECTOR_DIR = os.getenv("VECTOR_DIR", "vector_store")
EMB_MODEL  = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
TOP_K      = int(os.getenv("TOP_K", "5"))
# Ngưỡng coi là "không có context tốt" (chỉ dùng khi gọi ...with_score); tuỳ vectorstore mà thang điểm khác nhau.
RAG_MAX_DISTANCE = float(os.getenv("RAG_MAX_DISTANCE", "0.6"))

SYSTEM_POLICY = """
Bạn là một lãnh đạo cấp cao phụ trách hoạt động chiến lược của tập đoàn bảo hiểm phi nhân thọ,
đồng thời có nhiều kinh nghiệm điều hành và quản lý chiến lược ở các tập đoàn đa ngành nghề.

Nguyên tắc nền tảng:
- Ưu tiên dùng kiến thức nội bộ (nếu có context), nhưng không hiển thị nguồn/citation hay đường dẫn tệp cho người dùng;
  hãy trình bày như "theo kinh nghiệm và hiểu biết của tôi".
- Nếu không có nội dung khớp trong tài liệu nội bộ, hãy trả lời dựa trên kiến thức tổng hợp của bạn.
- Có thể gợi ý người dùng bổ sung tài liệu khi cần.
- Chỉ trả lời về bảo hiểm khi người hỏi đề cập; nếu không thì hỗ trợ chiến lược tổng quan.
- Trình bày mạch lạc, ưu tiên bullet; khi phù hợp có thể phân cấp ý tưởng theo Bloom 1–6.
- Nhấn mạnh giải pháp thực tiễn có thể áp dụng ngay cho phòng Chiến lược.
"""

def _load_vs():
    embeddings = HuggingFaceEmbeddings(model_name=EMB_MODEL)
    return Chroma(persist_directory=VECTOR_DIR, embedding_function=embeddings)

def retrieve_with_scores(query: str, k: int = TOP_K):
    vs = _load_vs()
    try:
        return vs.similarity_search_with_score(query, k=k)
    except Exception:
        # Fallback nếu backend không hỗ trợ with_score
        docs = vs.similarity_search(query, k=k)
        # giả lập score thấp (tức là tốt) để vẫn coi là có context
        return [(d, 0.3) for d in docs]

def _build_context(hits) -> str:
    # Không đính kèm đường dẫn/source để tránh lộ ra UI
    return "\n\n".join(f"[{i}] {d.page_content}" for i, d in enumerate(hits, 1))

def answer(query: str, allow_fallback: bool = True, min_rel: float = 0.6, k: int = TOP_K):
    vs = _load_vs()
    try:
        docs_scores = vs.similarity_search_with_score(query, k=k)
    except Exception:
        docs = vs.similarity_search(query, k=k)
        docs_scores = [(d, 0.3) for d in docs]  # giả score

    hits = [d for d, dist in docs_scores if dist <= min_rel]

    if hits:
        context = "\n\n".join(f"[{i}] {d.page_content}" for i, d in enumerate(hits, 1))
        messages = [
            {"role": "system", "content": SYSTEM_POLICY},
            {"role": "user", "content":
                "Ưu tiên NGỮ CẢNH NỘI BỘ sau. Nếu chưa đủ, có thể bổ sung kiến thức tổng hợp. "
                "Trình bày mạch lạc, ưu tiên bullet.\n\n"
                f"NGỮ CẢNH NỘI BỘ:\n{context}\n\nCÂU HỎI: {query}"},
        ]
        text = call_llm(messages)
        return text, []
    else:
        if not allow_fallback:
            return "Không có trong tài liệu đã đánh chỉ mục.", []
        messages = [
            {"role": "system", "content": SYSTEM_POLICY},
            {"role": "user", "content":
                "Hiện không có ngữ cảnh nội bộ phù hợp. "
                "Hãy trả lời dựa trên kiến thức tổng hợp của bạn và trình bày ngắn gọn theo bullet. "
                f"CÂU HỎI: {query}"},
        ]
        text = call_llm(messages)
        return text, []
