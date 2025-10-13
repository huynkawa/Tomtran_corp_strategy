# src/rag_chain.py
from src.reranker import rerank
import os
from openai import OpenAI
import time

# =========================
# === RETRIEVAL (giữ nguyên API)
# =========================
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


# =========================
# === ASSISTANTS HELPERS (mới)
# =========================
def _extract_assistant_text(client: OpenAI, thread_id: str) -> str:
    """
    Lấy toàn bộ text từ các message role=assistant trong thread (theo thứ tự tăng dần),
    ghép lại thành 1 chuỗi. Ưu tiên blk.type == 'text'.
    """
    try:
        msgs = client.beta.threads.messages.list(thread_id=thread_id, order="asc")
    except Exception as e:
        return f"ERROR::messages_list::{e}"

    out = []
    for m in msgs.data:
        if getattr(m, "role", "") != "assistant":
            continue
        for blk in getattr(m, "content", []) or []:
            if getattr(blk, "type", None) == "text":
                txt = getattr(getattr(blk, "text", None), "value", "") or ""
                if txt:
                    out.append(txt)
    return "\n".join(out).strip() or "(Không có nội dung Assistant trả về)"


def _run_assistant_safe(client: OpenAI, thread_id: str, assistant_id: str, timeout: int = 60) -> str:
    """
    Tạo run, poll an toàn tới khi:
      - completed: trả text
      - requires_action: trả chuỗi đặc biệt để UI biết phải xử lý
      - failed/cancelled/expired: trả ERROR::<code>::<message>
      - timeout: trả lỗi timeout
    """
    try:
        run = client.beta.threads.runs.create(thread_id=thread_id, assistant_id=assistant_id)
        print(f"[AssistantAPI] 🚀 Run created: {run.id} | Assistant: {assistant_id}")
    except Exception as e:
        return f"ERROR::run_create::{e}"

    start = time.time()
    while True:
        try:
            run_status = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)
        except Exception as e:
            return f"ERROR::run_retrieve::{e}"

        status = getattr(run_status, "status", "")
        if status in ("completed", "failed", "requires_action", "cancelled", "expired"):
            # kết thúc poll
            if status == "completed":
                return _extract_assistant_text(client, thread_id)

            if status == "requires_action":
                # Bạn đang bật tool (retrieval/code_interpreter/functions) mà chưa submit tool_outputs.
                # Trả về chuỗi đặc biệt để UI layer hiển thị rõ.
                return "__REQUIRES_ACTION__"

            if status in ("failed", "cancelled", "expired"):
                # Lấy last_error và tách code/message an toàn (object hoặc dict đều OK)
                err = getattr(run_status, "last_error", None)

                if isinstance(err, dict):
                    code = err.get("code", status)
                    msg  = err.get("message", "Unknown")
                else:
                    # err có thể là object kiểu LastError
                    code = getattr(err, "code", None) or status
                    # message có thể ở thuộc tính .message hoặc args[0], fallback str(err)
                    msg  = getattr(err, "message", None)
                    if not msg and getattr(err, "args", None):
                        msg = err.args[0]
                    if not msg:
                        msg = str(err) if err is not None else "Unknown"
                print(f"[AssistantAPI] ❌ Run {status} | code={code} | msg={msg}")
                return f"ERROR::{code}::{msg}"


        if time.time() - start > timeout:
            return "ERROR::timeout::Hết thời gian chờ phản hồi từ Assistant"

        time.sleep(0.5)


# =========================
# === PUBLIC API (giữ nguyên tên)
# =========================
def rag_answer(query, retriever_or_db, client=None, use_fallback=True, threshold=0.2, k=12, debug=False):
    """
    Trả lời câu hỏi dựa trên tài liệu (RAG) bằng Assistant API.
    Luôn trả về dict chuẩn:
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

    # Reuse 1 thread cho session process (như bản cũ)
    if not hasattr(rag_answer, "_thread") or rag_answer._thread is None:
        try:
            rag_answer._thread = client.beta.threads.create()
        except Exception as e:
            return {
                "answer": f"ERROR::thread_create::{e}",
                "source": "none",
                "ctx_text": ctx_text if ok else "",
                "docs": docs if ok else [],
            }

    # --- Nếu có context nội bộ ---
    if ok:
        # Rerank top-3 để rút gọn context
        try:
            reranked = rerank(query, docs, top_n=3, return_scores=True)
            docs = [d for d, _ in reranked]
            ctx_text = "\n\n".join(d.page_content for d in docs)
            if debug:
                print(f"[rerank] Query: {query}")
                for d, score in reranked:
                    print(f"   - Score={score:.4f} | Source={d.metadata.get('source', 'unknown')}")
        except Exception as e:
            print("[rag_chain] ⚠️ rerank error:", e)
            # vẫn tiếp tục với ctx_text ban đầu

        prompt = f"""
Dữ liệu nội bộ (context):
{ctx_text}

Câu hỏi của người dùng:
{query}

Hãy trả lời dựa trên dữ liệu nội bộ nếu có.
Nếu không có, hãy dùng kiến thức chung (và nêu rõ nguồn).
""".strip()

        # Đẩy message vào thread
        try:
            client.beta.threads.messages.create(
                thread_id=rag_answer._thread.id,
                role="user",
                content=prompt
            )
        except Exception as e:
            return {
                "answer": f"ERROR::messages_create::{e}",
                "source": "internal",
                "ctx_text": ctx_text,
                "docs": docs,
            }

        # Chạy assistant với poll an toàn
        answer_text = _run_assistant_safe(client, rag_answer._thread.id, assistant_id)

        return {
            "answer": answer_text,
            "source": "internal",
            "ctx_text": ctx_text,
            "docs": docs,
        }

    # --- Nếu không có context và cho phép fallback (kiến thức chung) ---
    if use_fallback:
        try:
            client.beta.threads.messages.create(
                thread_id=rag_answer._thread.id,
                role="user",
                content=query
            )
        except Exception as e:
            return {
                "answer": f"ERROR::messages_create::{e}",
                "source": "general",
                "ctx_text": "",
                "docs": [],
            }

        answer_text = _run_assistant_safe(client, rag_answer._thread.id, assistant_id)

        return {
            "answer": answer_text,
            "source": "general",
            "ctx_text": "",
            "docs": [],
        }

    # --- Nếu không có context và không fallback ---
    return {
        "answer": "⚠️ Không tìm thấy dữ liệu phù hợp và không bật fallback.",
        "source": "none",
        "ctx_text": "",
        "docs": [],
    }
