#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== ComfyUI SVD Image-to-Video Startup ==="

echo "Checking NVIDIA GPU..."
nvidia-smi || echo "WARNING: nvidia-smi not available, running on CPU"

echo "Checking CUDA availability in PyTorch..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}')" 2>/dev/null || echo "PyTorch not installed yet"

echo "Installing dependencies..."
if [ -f requirements-cpu.txt ] && python -c "import torch; exit(0 if not torch.cuda.is_available() else 1)" 2>/dev/null; then
    pip install -r requirements-cpu.txt
else
    pip install -r requirements.txt
fi

echo "Installing ComfyUI-SVD custom node..."
if [ -d "ComfyUI/custom_nodes/ComfyUI-SVD" ]; then
    echo "ComfyUI-SVD node already installed"
else
    cd ComfyUI/custom_nodes && git clone --depth 1 https://github.com/kijai/ComfyUI-SVD.git && cd -
fi

echo "Pre-downloading model (if not cached)..."
python -c "
from diffusers import StableVideoDiffusionPipeline
import torch
import os
cache_dir = os.environ.get('HF_HOME', './models')
print(f'Cache dir: {cache_dir}')
pipe = StableVideoDiffusionPipeline.from_pretrained(
    'stabilityai/stable-video-diffusion-img2vid-xt',
    torch_dtype=torch.float16,
    variant='fp16',
    cache_dir=cache_dir,
    local_files_only=False,
)
print('Model ready')
" || echo "Model download failed or already cached"

mkdir -p ./models ./outputs ./temp ./workflows

echo "Starting ComfyUI SVD application..."
exec python app.py
