import os

# 1. Nếu chạy trên Streamlit Cloud → ưu tiên st.secrets
try:
    import streamlit as st
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
        print("🔑 DEBUG: Đang chạy trên Cloud → load key từ st.secrets")
    else:
        raise ImportError("Không có OPENAI_API_KEY trong st.secrets")
except Exception:
    # 2. Nếu chạy local → ưu tiên file .env.active
    from dotenv import load_dotenv
    if os.path.exists(".env.active"):
        load_dotenv(".env.active")
        print("🔑 DEBUG: Đang chạy local → load key từ .env.active")
    else:
        load_dotenv()
        print("🔑 DEBUG: Đang chạy local → load key từ .env")
