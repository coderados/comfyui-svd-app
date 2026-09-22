# ComfyUI SVD Image-to-Video Generation App

Containerized Stable Video Diffusion (SVD-XT) application built with **ComfyUI** workflow engine, REST API, and Gradio Web UI.

## Features

- **ComfyUI Workflow Engine** - Node-based SVD pipeline using ComfyUI custom nodes
- **Stable Video Diffusion XT** (25 frames, 1024x576)
- **REST API** (`/generate`) for programmatic access
- **Gradio Web UI** (`/ui`) for interactive use
- **ComfyUI Workflow API** (`/workflow`) to retrieve the workflow JSON
- **Unrestricted/Unfiltered** - Safety checker disabled by default
- **Optimized for VRAM** - CPU offload, VAE slicing/tiling, xformers
- **Queue-based generation** - Handles concurrent requests

## Quick Start

### Using Docker Compose (Recommended)

```bash
# Build and start
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### Pre-download Model (Optional)

```bash
docker-compose --profile download up model-downloader
```

### Manual Docker Build

```bash
docker build -t comfyui-svd-app .
docker run -d \
  --name comfyui-svd-app \
  --gpus all \
  -p 7860:7860 \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/temp:/app/temp \
  comfyui-svd-app
```

### Local Development

```bash
pip install -r requirements.txt
python app.py
```

## Access Points

| Interface | URL |
|-----------|-----|
| Gradio Web UI | http://localhost:7860/ui |
| REST API Docs | http://localhost:7860/docs |
| Health Check | http://localhost:7860/health |
| ComfyUI Workflow | http://localhost:7860/workflow |
| Queue Status | http://localhost:7860/queue/status |

## API Usage

### Generate Video

```bash
curl -X POST "http://localhost:7860/generate" \
  -F "file=@input_image.jpg" \
  -F "num_frames=14" \
  -F "num_inference_steps=25" \
  -F "motion_bucket_id=127" \
  -F "fps=7" \
  -F "disable_safety=true" \
  -F "enable_comfyui=false"
```

Response:
```json
{
  "job_id": "abc123...",
  "status": "completed",
  "message": "Video generated successfully via ComfyUI SVD",
  "video_url": "/outputs/abc123.mp4",
  "workflow_id": "abc123..."
}
```

### Download Video

```bash
curl -O "http://localhost:7860/outputs/abc123.mp4"
```

### Get ComfyUI Workflow

```bash
curl http://localhost:7860/workflow
```

## Parameters

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| `num_frames` | 14-25 | 14 | Number of frames (XT model supports 25) |
| `num_inference_steps` | 10-50 | 25 | Denoising steps (more = better quality, slower) |
| `motion_bucket_id` | 1-255 | 127 | Motion intensity (1=static, 255=high motion) |
| `fps` | 1-30 | 7 | Output video frame rate |
| `noise_aug_strength` | 0.0-1.0 | 0.02 | Noise added to input image |
| `decode_chunk_size` | 1-16 | 8 | VAE decode chunk size (lower = less VRAM) |
| `seed` | int | random | Random seed for reproducibility |
| `disable_safety` | bool | true | Disable safety checker |
| `enable_comfyui` | bool | false | Use ComfyUI workflow engine |

## ComfyUI Integration

The app integrates with ComfyUI through:

1. **ComfyUI-SVD Custom Node** (`ComfyUI/custom_nodes/ComfyUI-SVD/`) - Provides `StableVideoDiffusionSampler`, `SVDPipelineLoader`, and `ImageToVideoComfy` nodes
2. **Workflow JSON** (`workflows/svd_image_to_video.json`) - ComfyUI workflow definition with nodes for KSampler, VAEEncode, CheckpointLoader, VideoCombine, etc.
3. **Node Definitions** (`workflows/svd_nodes.json`) - Custom node schemas for SVD operations

### Workflow Nodes

- `EmptyLatentImage` - Creates latent tensor (1024x576)
- `CLIPTextEncode` - Text conditioning (positive/negative)
- `CheckpointLoaderSimple` - Loads SVD model
- `KSampler` - Diffusion sampler with SVD parameters
- `VAEEncode` - Encodes input image to latent space
- `VAEDecode` - Decodes latent to video frames
- `VideoCombine` - Combines frames into video output
- `LoadImage` - Loads input image
- `ImageScale` - Resizes image to 1024x576

## VRAM Requirements

| Frames | Resolution | Min VRAM (with optimizations) |
|--------|------------|------------------------------|
| 14 | 1024x576 | ~6 GB |
| 25 | 1024x576 | ~10 GB |

If OOM: reduce `num_frames`, `decode_chunk_size`, or enable `cpu_offload`.

## Model Variants

- **SVD-XT** (`stable-video-diffusion-img2vid-xt`): 25 frames, higher quality
- **SVD** (`stable-video-diffusion-img2vid`): 14 frames, faster

Change `MODEL_ID` in `.env` or `app.py` to switch.

## Directory Structure

```
svd-app/
├── app.py                  # Main FastAPI + Gradio application
├── workflow.json           # ComfyUI workflow definition
├── svd_nodes.json          # Custom node definitions
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container configuration
├── docker-compose.yml      # Docker compose
├── start.sh               # Startup script
├── README.md              # This file
├── models/                # Hugging Face model cache
├── outputs/               # Generated videos
├── temp/                  # Temporary files
└── workflows/             # ComfyUI workflow JSONs
    ├── svd_image_to_video.json
    └── svd_nodes.json
```

## Troubleshooting

### Out of Memory
```bash
# Reduce decode_chunk_size in API request
# Or edit app.py to enable sequential CPU offload:
pipe.enable_sequential_cpu_offload()
```

### xformers not working
```bash
# Install with: pip install xformers --index-url https://download.pytorch.org/whl/cu121
# Or disable in app.py: pipe.enable_xformers_memory_efficient_attention()
```

### Slow Generation
- First run downloads model (~10GB)
- Use `num_inference_steps=15-20` for faster preview
- Ensure GPU is not throttling (check `nvidia-smi`)

## ComfyUI Manager Integration

To install this app using ComfyUI-Manager:
1. Open ComfyUI-Manager
2. Search for "ComfyUI-SVD" custom node
3. Install the node
4. Load the workflow JSON from `workflows/svd_image_to_video.json`

## License

Stable Video Diffusion model: [Stability AI License](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/blob/main/LICENSE)
Code: MIT License
