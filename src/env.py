import os
from pathlib import Path

print("=== DEBUG: Bắt đầu load OPENAI_API_KEY ===")

# Ưu tiên: lấy key từ Streamlit Cloud (Secrets Manager)
try:
    import streamlit as st
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
        print("✅ DEBUG: Key loaded từ Streamlit secrets")
    else:
        print("⚠️ DEBUG: Streamlit secrets không có OPENAI_API_KEY")
except Exception as e:
    print("⚠️ DEBUG: Không chạy trên Streamlit hoặc không import được:", e)

# Nếu chưa có, load từ .env.active hoặc .env (ép UTF-8 để tránh lỗi Windows cp1258)
if "OPENAI_API_KEY" not in os.environ:
    from dotenv import dotenv_values, load_dotenv
    root_dir = Path(__file__).resolve().parent.parent

    env_active = root_dir / ".env.active"
    env_file = root_dir / ".env"

    if env_active.exists():
        values = dotenv_values(env_active, encoding="utf-8")
        for k, v in values.items():
            if v is not None:
                os.environ[k] = v
        print("✅ DEBUG: Key loaded từ .env.active (UTF-8 forced)")
    elif env_file.exists():
        load_dotenv(env_file)
        print("✅ DEBUG: Key loaded từ .env")
    else:
        print("⚠️ DEBUG: Không tìm thấy .env.active hoặc .env")

# Kiểm tra key cuối cùng
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("❌ ERROR: OPENAI_API_KEY chưa được set. Vui lòng cấu hình trong .env hoặc Streamlit secrets.")
else:
    print("🔑 DEBUG: OPENAI_API_KEY =", api_key[:10], "...(ẩn phần còn lại)")

print("=== DEBUG: Kết thúc load OPENAI_API_KEY ===")
