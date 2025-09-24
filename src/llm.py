import os
import time
from typing import List, Dict
import src.env

LLM_MODE = os.getenv("LLM_MODE", "OPENAI").upper()


def call_llm(messages: List[Dict[str, str]]) -> str:
    """
    messages: [{"role": "system"|"user"|"assistant", "content": "..."}]
    return: model output text
    """
    if LLM_MODE == "OPENAI":
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1"),
            messages=messages,
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
        )
        return resp.choices[0].message.content or ""

    elif LLM_MODE == "ASSISTANT":
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        assistant_id = os.getenv("ASSISTANT_ID")
        if not assistant_id:
            raise RuntimeError("Thiếu ASSISTANT_ID trong .env.active")

        # 1. Tạo thread
        thread = client.beta.threads.create()

        # 2. Gửi các message user vào thread
        for msg in messages:
            if msg["role"] == "user":
                client.beta.threads.messages.create(
                    thread_id=thread.id,
                    role="user",
                    content=msg["content"]
                )

        # 3. Tạo run cho assistant
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant_id
        )

        # 4. Poll kết quả
        while True:
            run_status = client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id
            )
            if run_status.status == "completed":
                break
            elif run_status.status in ("failed", "expired"):
                raise RuntimeError(f"Run failed: {run_status.status}")
            time.sleep(1)

        # 5. Lấy message cuối cùng từ assistant
        msgs = client.beta.threads.messages.list(thread_id=thread.id)
        for m in msgs.data:
            if m.role == "assistant":
                if m.content and len(m.content) > 0:
                    return m.content[0].text.value
        return ""

    elif LLM_MODE == "OLLAMA":
        import requests, json
        base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        model = os.getenv("OLLAMA_MODEL", "llama3:instruct")
        prompt = "\n\n".join([m["content"] for m in messages if m["role"] in ("system","user")])
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": float(os.getenv("LLM_TEMPERATURE", "0.2"))}
        }
        r = requests.post(f"{base}/api/generate", json=payload, timeout=600)
        r.raise_for_status()
        return r.json().get("response", "").strip()

    else:
        raise ValueError("LLM_MODE phải là OPENAI, ASSISTANT hoặc OLLAMA")
