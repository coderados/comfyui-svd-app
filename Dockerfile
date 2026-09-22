FROM nvidia/cuda:12.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3.10-venv \
    git wget curl ffmpeg libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.10 /usr/bin/python && \
    ln -s /usr/bin/python3.10 /usr/bin/python3

RUN python -m pip install --upgrade pip setuptools wheel

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN chmod +x start.sh

RUN mkdir -p /app/models /app/outputs /app/temp /app/workflows

ENV HF_HOME=/app/models \
    TRANSFORMERS_CACHE=/app/models \
    DIFFUSERS_CACHE=/app/models \
    COMFYUI_PATH=/app/ComfyUI \
    MODEL_ID=stabilityai/stable-video-diffusion-img2vid-xt

EXPOSE 7860 8000

CMD ["./start.sh"]
