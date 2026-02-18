FROM python:3.11-slim AS build

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

RUN pip install fastapi[standard]==0.104.1 \
    uvicorn[standard]==0.24.0 \
    python-dotenv==1.0.0 \
    python-multipart==0.0.6 \
    python-dotenv \
    opencv-python \
    numpy==1.24.4

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

COPY . .

EXPOSE 443

CMD ["python3", "main.py"]  