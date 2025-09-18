FROM python:3.11-slim

# Cài gói cần thiết khi build sentence-transformers/FAISS
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
ENV PYTHONPATH=/app

# Cổng Streamlit
EXPOSE 8501

# Chạy: build index (nếu có dữ liệu mount vào) rồi khởi động app
CMD bash -lc "python -m src.build_index || true && streamlit run app.py --server.address=0.0.0.0 --server.port=8501"
