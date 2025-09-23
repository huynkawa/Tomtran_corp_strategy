# src/rag_chain.py
def rag_answer(query, retriever, llm, client=None, use_fallback=True, threshold=0.2, k=6):
    """
    Trả lời câu hỏi dựa trên tài liệu (RAG).
    Nếu không tìm thấy tài liệu phù hợp, fallback sang GPT tổng quát.
    """

    docs = []
    try:
        # Lấy tài liệu kèm điểm số
        pairs = retriever.vectorstore.similarity_search_with_relevance_scores(query, k=k)
        docs = [d for (d, s) in pairs if (s is not None and s >= threshold)]
        if not docs:  # nếu không đủ điểm thì lấy tất cả
            docs = [d for (d, _) in pairs]
    except Exception:
        try:
            docs = retriever.get_relevant_documents(query)
        except Exception:
            docs = []

    # --- 2. Nếu có context → trả lời từ RAG ---
    if docs:
        context = "\n\n".join([d.page_content for d in docs])
        prompt = f"""Bạn là một trợ lý phân tích báo cáo tài chính.
Hãy trả lời dựa trên Context dưới đây:
- Nếu là số liệu, luôn nêu rõ năm/quý (Số cuối năm, Số đầu năm).
- Luôn ghi rõ đơn vị (VND, USD, %, ...).
- Nếu lấy từ bảng, hãy nói rõ tiêu đề cột và tiêu đề dòng.
- Nếu có nhiều giá trị (ví dụ: cuối năm và đầu năm), hãy liệt kê cả hai.
- Nếu không có dữ liệu phù hợp, hãy nói: 'Không tìm thấy trong tài liệu.'.

Context:
{context}

Câu hỏi: {query}
"""
        result = llm.invoke(prompt)

        if hasattr(result, "content"):
            return "🏠 " + result.content  # icon ngôi nhà = từ dữ liệu nội bộ
        return "🏠 " + str(result)

    # --- 3. Nếu không có context → fallback GPT ---
    if use_fallback and client:
        try:
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Bạn là một trợ lý hữu ích, trả lời bằng kiến thức tổng quát."},
                    {"role": "user", "content": query}
                ]
            )
            return "🌐 " + completion.choices[0].message.content  # icon quả cầu = kiến thức chung
        except Exception as e:
            return f"Lỗi khi gọi GPT fallback: {e}"

    # --- 4. Nếu không có gì ---
    return "Không có trong tài liệu đã đánh chỉ mục."
