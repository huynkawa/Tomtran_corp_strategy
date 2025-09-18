import os
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
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=messages,
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
        )
        return resp.choices[0].message.content or ""

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
        raise ValueError("LLM_MODE phải là OPENAI hoặc OLLAMA")