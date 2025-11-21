# Usar imagen base con CUDA para soporte GPU
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Configurar timezone para instalación no-interactiva
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Bogota

# Instalar Python 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3-pip \
    gcc \
    build-essential \
    libffi-dev \
    tesseract-ocr \
    tesseract-ocr-spa \
    poppler-utils \
    libtesseract-dev \
    libleptonica-dev \
    libjpeg-dev \
    zlib1g-dev \
    libpango-1.0-0 \
    libpangoft2-1.0-0 \
    libcairo2 \
    libgirepository1.0-dev \
    && rm -rf /var/lib/apt/lists/*

# Crear symlinks para python3.11
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Configurar variables de entorno para CUDA
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV CUDA_VISIBLE_DEVICES=0

WORKDIR /app

COPY requirements.txt .

# Instalar PyTorch con soporte CUDA
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
    pip3 install --no-cache-dir -r requirements.txt

COPY ./app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]