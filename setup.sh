#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== ComfyUI SVD Image-to-Video Setup ==="

# Clone ComfyUI if not present
if [ ! -d "ComfyUI" ]; then
    echo "Cloning ComfyUI..."
    git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git
fi

# Clone ComfyUI-SVD custom node if not present
if [ ! -d "ComfyUI/custom_nodes/ComfyUI-SVD" ]; then
    echo "Installing ComfyUI-SVD custom node..."
    cd ComfyUI/custom_nodes
    git clone --depth 1 https://github.com/kijai/ComfyUI-SVD.git
    cd ../..
fi

# Clone ComfyUI Manager if not present
if [ ! -d "ComfyUI/custom_nodes/comfyui-manager" ]; then
    echo "Installing ComfyUI-Manager..."
    cd ComfyUI/custom_nodes
    git clone --depth 1 https://github.com/ltdrdata/ComfyUI-Manager.git
    cd ../..
fi

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Download SVD model
echo "Pre-downloading SVD model..."
python -c "
from diffusers import StableVideoDiffusionPipeline
import torch
import os
cache_dir = os.environ.get('HF_HOME', './models')
pipe = StableVideoDiffusionPipeline.from_pretrained(
    'stabilityai/stable-video-diffusion-img2vid-xt',
    torch_dtype=torch.float16,
    variant='fp16',
    cache_dir=cache_dir,
    local_files_only=False,
)
print('Model ready')
" || echo "Model download skipped (may already be cached)"

# Setup workflows
mkdir -p workflows models outputs temp

echo "=== Setup Complete ==="
echo "To start the app: python app.py"
echo "Or with Docker: docker-compose up -d --build"
