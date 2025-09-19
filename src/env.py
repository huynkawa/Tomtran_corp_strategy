import os

# 1. Ưu tiên lấy API key từ Streamlit Cloud Secrets
try:
    import streamlit as st
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
        os.environ["OPENAI_MODEL"] = st.secrets.get("OPENAI_MODEL", "gpt-4o-mini")
        print("🔑 DEBUG: Đang chạy trên Streamlit Cloud → dùng st.secrets")
except Exception:
    pass

# 2. Khi chạy local mới dùng dotenv để load file .env.active
try:
    from dotenv import load_dotenv
    if load_dotenv(dotenv_path=".env.active"):
        print("🔑 DEBUG: Đang chạy local → đã load key từ .env.active")
    else:
        print("⚠️ DEBUG: Không tìm thấy file .env.active")
except ModuleNotFoundError:
    # Không có python-dotenv (trên Cloud) cũng không sao
    print("⚠️ DEBUG: python-dotenv chưa cài, bỏ qua bước load .env.active")
    pass
