from src.reranker import rerank
import os
from openai import OpenAI
import time  # cần để dùng sleep

def retrieve_context_any(db_or_retriever, query, k=12, min_relev=0.2):
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


def wait_for_completion(client, thread_id, run_id, timeout=60):
    """Poll Assistant API cho đến khi có kết quả hoặc hết timeout"""
    start = time.time()
    while time.time() - start < timeout:
        run_status = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
        if run_status.status == "completed":
            messages = client.beta.threads.messages.list(thread_id=thread_id)
            # Lấy message cuối cùng của assistant
            for m in messages.data:
                if m.role == "assistant":
                    return m.content[0].text.value
            return "(Không nhận được message từ Assistant)"
        elif run_status.status in ["failed", "cancelled", "expired"]:
            return f"(Assistant run error: {run_status.status})"
        time.sleep(1)
    return "(Timeout khi chờ Assistant trả lời)"


def rag_answer(query, retriever_or_db, client=None, use_fallback=True, threshold=0.2, k=12, debug=False):
    """
    Trả lời câu hỏi dựa trên tài liệu (RAG) bằng Assistant API.
    Trả về dict:
      {
        "answer": str,
        "source": "internal" | "general" | "none",
        "ctx_text": str,
        "docs": list
      }
    """
    ctx_text, docs, ok = retrieve_context_any(retriever_or_db, query, k=k, min_relev=threshold)

    api_key = os.getenv("OPENAI_API_KEY")
    assistant_id = os.getenv("ASSISTANT_ID")
    if not assistant_id:
        return {
            "answer": "❌ Thiếu ASSISTANT_ID trong .env.active",
            "source": "none",
            "ctx_text": ctx_text if ok else "",
            "docs": docs if ok else [],
        }

    client = client or OpenAI(api_key=api_key)

    if not hasattr(rag_answer, "_thread"):
        rag_answer._thread = client.beta.threads.create()

    # --- Nếu có context nội bộ ---
    if ok:
        reranked = rerank(query, docs, top_n=3, return_scores=True)
        docs = [d for d, _ in reranked]
        ctx_text = "\n\n".join(d.page_content for d in docs)

        if debug:
            print(f"[rerank] Query: {query}")
            for d, score in reranked:
                print(f"   - Score={score:.4f} | Source={d.metadata.get('source', 'unknown')}")

        # Soạn prompt cho Assistant
        prompt = f"""
Dữ liệu nội bộ (context):
{ctx_text}

Câu hỏi của người dùng:
{query}

Hãy trả lời dựa trên dữ liệu nội bộ nếu có.
Nếu không có, hãy dùng kiến thức chung (và nêu rõ nguồn).
""".strip()

        # Tạo message trong thread
        client.beta.threads.messages.create(
            thread_id=rag_answer._thread.id,
            role="user",
            content=prompt
        )

        # Tạo run
        run = client.beta.threads.runs.create(
            thread_id=rag_answer._thread.id,
            assistant_id=assistant_id
        )
        print(f"[AssistantAPI] 🚀 Run created: {run.id} | Assistant: {assistant_id}")

        # Đợi Assistant trả lời
        answer_text = wait_for_completion(client, rag_answer._thread.id, run.id)

        print(f"[AssistantAPI] ✅ Assistant trả lời xong | Run={run.id}")
        print(f"[AssistantAPI] ✍️ Answer (preview): {answer_text[:200]}...")

        return {
            "answer": answer_text,
            "source": "internal",
            "ctx_text": ctx_text,
            "docs": docs,
        }

    # --- Nếu không có context và cho phép fallback ---
    if use_fallback:
        client.beta.threads.messages.create(
            thread_id=rag_answer._thread.id,
            role="user",
            content=query
        )

        run = client.beta.threads.runs.create(
            thread_id=rag_answer._thread.id,
            assistant_id=assistant_id
        )
        print(f"[AssistantAPI] 🚀 Run created (fallback): {run.id} | Assistant: {assistant_id}")

        answer_text = wait_for_completion(client, rag_answer._thread.id, run.id)
