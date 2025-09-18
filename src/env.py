# src/env.py
import os
from dotenv import load_dotenv

# Ưu tiên .env.active nếu tồn tại
if os.path.isfile(".env.active"):
    load_dotenv(".env.active", override=True)
else:
    load_dotenv()
