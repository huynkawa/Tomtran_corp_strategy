# src/chat_saver.py
# -*- coding: utf-8 -*-
"""
Module: chat_saver.py
Tự động lưu và đọc lịch sử hội thoại ra file JSON
"""

import os
import json
from datetime import datetime

CHAT_DIR = "chat_history"

def ensure_dir():
    """Tạo thư mục lưu chat nếu chưa tồn tại"""
    os.makedirs(CHAT_DIR, exist_ok=True)

def get_chat_path(chat_name):
    """Tạo đường dẫn file theo tên hội thoại"""
    safe_name = chat_name.replace(" ", "_").replace("#", "")
    return os.path.join(CHAT_DIR, f"{safe_name}.json")

def save_chat(chat_name, messages):
    """Lưu toàn bộ hội thoại vào file JSON"""
    ensure_dir()
    data = {
        "chat_name": chat_name,
        "updated_at": datetime.now().isoformat(),
        "messages": messages
    }
    path = get_chat_path(chat_name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path

def load_all_chats():
    """Đọc tất cả các hội thoại đã lưu"""
    ensure_dir()
    chats = []
    for f in os.listdir(CHAT_DIR):
        if f.endswith(".json"):
            path = os.path.join(CHAT_DIR, f)
            try:
                with open(path, "r", encoding="utf-8") as file:
                    data = json.load(file)
                    chats.append(data)
            except Exception as e:
                print(f"Lỗi đọc {f}: {e}")
    # Sắp xếp theo thời gian cập nhật mới nhất
    chats.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    return chats
