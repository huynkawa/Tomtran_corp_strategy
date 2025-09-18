# src/app.py
import os
import streamlit as st
import src.env  # nạp .env.active / .env

# Thử import SDK mới (OpenAI>=1.0.0)
try:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    USE_CLIENT = True
except ImportError:
    import openai
    client = None
    USE_CLIENT = False

from src.prompt_loader import load_prompts, render_system_prompt, list_profiles
from langchain_community.vectorstores import Chroma
from src.config import make_embeddings


def main():
    st.set_page_config(page_title="TOMTRANCHATBOT", layout="wide")
    st.title("TOMTRANCHATBOT")

    cfg = load_prompts("prompts/prompts.yaml")
    profile_map = list_profiles(cfg)
    keys = list(profile_map.keys())
    default_idx = keys.index("base") if "base" in keys else 0
    selected_key = st.sidebar.selectbox("Prompt profile", keys, index=default_idx)

    temperature = st.sidebar.slider("Temperature", 0.0, 1.2, 0.3, 0.1)
    top_p = st.sidebar.slider("Top_p", 0.1, 1.0, 1.0, 0.1)
    fallback_general = st.sidebar.checkbox("Fallback GPT nếu không có tài liệu phù hợp", value=True)
    K = st.sidebar.slider("Số đoạn context (k)", 1, 12, 4, 1)
    MIN_RELEVANCE = st.sidebar.slider(
        "Ngưỡng điểm liên quan tối thiểu (0–1, cao = chặt)", 0.0, 1.0, 0.30, 0.05
    )

    system_prompt = render_system_prompt(cfg, selected_key)
    effective_profile = selected_key
    if selected_key == "rag" and fallback_general:
        effective_profile = "base" if "base" in keys else selected_key
        system_prompt = render_system_prompt(cfg, effective_profile)
        st.sidebar.info(
            "Profile 'rag' là RAG-only. Đã tạm dùng profile 'base' để cho phép fallback GPT."
        )

    with st.expander("🔧 System prompt đang dùng", expanded=False):
        st.code(system_prompt, language="markdown")

    @st.cache_resource
    def get_vectordb():
        vector_dir = os.getenv("VECTOR_DIR", "vector_store")
        return Chroma(
            persist_directory=vector_dir, embedding_function=make_embeddings()
        )

    def retrieve_context(db, query: str, k: int, threshold: float):
        try:
            pairs = db.similarity_search_with_relevance_scores(query, k=k)
            docs = [d for (d, s) in pairs if (s is not None and s >= threshold)]
            if not docs:
                docs = [d for (d, _) in pairs]
        except Exception:
            try:
                pairs = db.similarity_search_with_score(query, k=k)
                kept = [(d, s) for (d, s) in pairs if (s is not None and s <= threshold)]
                docs = [d for d, _ in kept] if kept else [d for (d, _) in pairs]
            except Exception:
                docs = db.similarity_search(query, k=k)

        if not docs:
            return "NO_CONTEXT", [], False

        ctx = "\n\n---\n".join(d.page_content for d in docs)
        return ctx, docs, True

    vectordb = get_vectordb()

    with st.expander("🧪 RAG diagnostics", expanded=False):
        try:
            emb = make_embeddings()
            st.write("Embedding class:", emb.__class__.__name__)
            vector_dir = os.getenv("VECTOR_DIR", "vector_store")
            st.write("Persist dir:", os.path.abspath(vector_dir))
            count = getattr(vectordb, "_collection").count()
            st.write("Vector count:", count)

            q = st.session_state.history[-1][1] if st.session_state.get("history") else ""
            if q:
                try:
                    pairs = vectordb.similarity_search_with_relevance_scores(q, k=5)
                    st.write("Top-5 relevance scores:", [float(s) for _, s in pairs])
                except Exception:
                    pairs = vectordb.similarity_search(q, k=5)
                    st.write(
                        "Top-5 (no scores):",
                        [p.page_content[:60] + "..." for p in pairs],
                    )
        except Exception as e:
            st.warning(f"Diag error: {e}")

    if "history" not in st.session_state:
        st.session_state.history = []

    user_msg = st.chat_input("Nhập câu hỏi…")
    if user_msg:
        st.session_state.history.append(("user", user_msg))

    if st.session_state.history:
        messages = [{"role": "system", "content": system_prompt}]
        debug_block = ""

        latest_query = st.session_state.history[-1][1] if user_msg else ""
        ctx_text, docs, ok = retrieve_context(vectordb, latest_query, K, MIN_RELEVANCE)

        if ok:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "CONTEXT (nguồn chính; KHÔNG lộ cho người dùng):\n"
                        f"{ctx_text}\n\n"
                        "HƯỚng dẫn: Ưu tiên CONTEXT làm sự thật. "
                        "Bạn CÓ THỂ bổ sung kiến thức tổng quát để hoàn thiện câu trả lời, "
                        "nhưng tuyệt đối không mâu thuẫn với CONTEXT."
                    ),
                }
            )
            debug_block = "\n".join(f"- {d.metadata.get('source')}" for d in docs)
        else:
            if fallback_general:
                messages.append(
                    {
                        "role": "system",
                        "content": (
                            "KHÔNG tìm thấy context phù hợp trong tài liệu đã đánh chỉ mục. "
                            "Hãy trả lời bằng kiến thức tổng quát của bạn (không cần trích dẫn), "
                            "và nêu rõ nếu câu hỏi có vẻ cần dữ liệu nội bộ."
                        ),
                    }
                )
                debug_block = "No relevant context found."
            else:
                st.session_state.history.append(
                    ("assistant", "Không có trong tài liệu đã đánh chỉ mục.")
                )
                for role, content in st.session_state.history:
                    with st.chat_message(role):
                        st.markdown(content)
                st.stop()

        for role, content in st.session_state.history:
            messages.append({"role": role, "content": content})

        # ✅ Gọi OpenAI API theo SDK phù hợp
        if USE_CLIENT:
            resp = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=messages,
                temperature=temperature,
                top_p=top_p,
            )
            assistant_msg = resp.choices[0].message.content or ""
        else:
            resp = openai.ChatCompletion.create(
                model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
                messages=messages,
                temperature=temperature,
                top_p=top_p,
            )
            assistant_msg = resp.choices[0].message["content"] or ""

        st.session_state.history.append(("assistant", assistant_msg))

        with st.expander("🔍 Debug context", expanded=False):
            st.markdown(debug_block or "—")

    for role, content in st.session_state.history:
        with st.chat_message(role):
            st.markdown(content)


if __name__ == "__main__":
    main()
