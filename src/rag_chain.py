# src/rag_chain.py

def retrieve_context_any(db_or_retriever, query, k=6, min_relev=0.2):
    """
    Lấy context từ vectordb hoặc retriever.
    Trả về (ctx_text, docs, ok).
    """
    docs = []
    try:
        # Ưu tiên vectordb (Chroma)
        if hasattr(db_or_retriever, "similarity_search_with_relevance_scores"):
            pairs = db_or_retriever.similarity_search_with_relevance_scores(query, k=k)
            docs = [d for (d, s) in pairs if (s is not None and s >= min_relev)]
            if not docs:  # nếu không đủ điểm thì lấy tất cả
                docs = [d for (d, _) in pairs]

        # Nếu không phải vectordb, fallback sang retriever
        elif hasattr(db_or_retriever, "get_relevant_documents"):
            docs = db_or_retriever.get_relevant_documents(query)

    except Exception as e:
        print("[rag_chain] ❌ retrieve_context_any error:", e)
        docs = []

    ok = bool(docs)
    ctx_text = "\n\n".join(d.page_content for d in docs) if docs else ""
    return ctx_text, docs, ok


def rag_answer(query, retriever_or_db, llm, client=None, use_fallback=True, threshold=0.2, k=6):
    """
    Trả lời câu hỏi dựa trên tài liệu (RAG).
    Trả về dict:
      {
        "answer": str,
        "source": "internal" | "general" | "none",
        "ctx_text": str,
        "docs": list
      }
    """
    ctx_text, docs, ok = retrieve_context_any(retriever_or_db, query, k=k, min_relev=threshold)

    # --- Nếu có context nội bộ ---
    if ok:
        prompt = f"""Bạn là một trợ lý phân tích báo cáo tài chính.
Hãy trả lời dựa trên Context dưới đây:
- Nếu là số liệu, luôn nêu rõ năm/quý (Số cuối năm, Số đầu năm).
- Luôn ghi rõ đơn vị (VND, USD, %, ...).
- Nếu lấy từ bảng, hãy nói rõ tiêu đề cột và tiêu đề dòng.
- Nếu có nhiều giá trị (ví dụ: cuối năm và đầu năm), hãy liệt kê cả hai.
- Nếu không có dữ liệu phù hợp, hãy nói: 'Không tìm thấy trong tài liệu.'.

Context:
{ctx_text}

Câu hỏi: {query}
"""
        result = llm.invoke(prompt)
        answer_text = result.content if hasattr(result, "content") else str(result)
        return {
            "answer": answer_text,
            "source": "internal",
            "ctx_text": ctx_text,
            "docs": docs,
        }

    # --- Nếu không có context và có fallback ---
    if use_fallback and client:
        try:
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Bạn là một trợ lý hữu ích, trả lời bằng kiến thức tổng quát."},
                    {"role": "user", "content": query}
                ]
            )
            return {
                "answer": completion.choices[0].message.content,
                "source": "general",
                "ctx_text": "",
                "docs": [],
            }
        except Exception as e:
            return {
                "answer": f"Lỗi khi gọi GPT fallback: {e}",
                "source": "none",
                "ctx_text": "",
                "docs": [],
            }

    # --- Nếu không có gì ---
    return {
        "answer": "Không có trong tài liệu đã đánh chỉ mục.",
        "source": "none",
        "ctx_text": "",
        "docs": [],
    }
