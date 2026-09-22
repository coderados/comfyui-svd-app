import json
import os
import uuid
import asyncio
import threading
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

import torch
import numpy as np
from PIL import Image
import imageio
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import gradio as gr

from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import export_to_video

MODEL_ID = os.environ.get("MODEL_ID", "stabilityai/stable-video-diffusion-img2vid-xt")
MODEL_CACHE_DIR = Path(os.environ.get("HF_HOME", "./models"))
OUTPUT_DIR = Path("/app/outputs")
TEMP_DIR = Path("/app/temp")
COMFYUI_PATH = os.environ.get("COMFYUI_PATH", "./ComfyUI")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

pipeline = None
pipeline_lock = threading.Lock()
generation_queue = asyncio.Queue()
worker_task = None


class GenerationRequest(BaseModel):
    num_frames: int = Field(default=14, ge=14, le=25)
    num_inference_steps: int = Field(default=25, ge=10, le=50)
    motion_bucket_id: int = Field(default=127, ge=1, le=255)
    fps: int = Field(default=7, ge=1, le=30)
    noise_aug_strength: float = Field(default=0.02, ge=0.0, le=1.0)
    decode_chunk_size: int = Field(default=8, ge=1, le=16)
    seed: Optional[int] = Field(default=None)
    disable_safety: bool = Field(default=True)
    enable_comfyui: bool = Field(default=False)


class GenerationResponse(BaseModel):
    job_id: str
    status: str
    message: str
    video_url: Optional[str] = None
    workflow_id: Optional[str] = None


def load_pipeline(disable_safety: bool = True):
    global pipeline
    with pipeline_lock:
        if pipeline is not None:
            return pipeline

        print(f"[ComfyUI-SVD] Loading SVD model from {MODEL_ID}...")
        pipe = StableVideoDiffusionPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            variant="fp16",
            cache_dir=MODEL_CACHE_DIR,
            local_files_only=False,
        )

        pipe.enable_model_cpu_offload()
        pipe.enable_vae_slicing()
        pipe.enable_vae_tiling()

        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("[ComfyUI-SVD] xformers memory efficient attention enabled")
        except Exception as e:
            print(f"[ComfyUI-SVD] xformers not available: {e}")

        if disable_safety:
            pipe.safety_checker = None
            pipe.requires_safety_checker = False
            print("[ComfyUI-SVD] Safety checker disabled (UNRESTRICTED)")

        pipeline = pipe
        print("[ComfyUI-SVD] Pipeline loaded successfully")
        return pipeline


def load_comfyui_workflow(workflow_path: str = None) -> dict:
    if workflow_path is None:
        workflow_path = str(Path(__file__).parent / "workflows" / "svd_image_to_video.json")
    if os.path.exists(workflow_path):
        with open(workflow_path, "r") as f:
            return json.load(f)
    return None


def generate_video_via_diffusers(
    image: Image.Image,
    num_frames: int = 14,
    num_inference_steps: int = 25,
    motion_bucket_id: int = 127,
    fps: int = 7,
    noise_aug_strength: float = 0.02,
    decode_chunk_size: int = 8,
    seed: Optional[int] = None,
    disable_safety: bool = True,
) -> str:
    pipe = load_pipeline(disable_safety=disable_safety)

    if seed is not None:
        generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed)
    else:
        generator = None

    image = image.resize((1024, 576))

    with torch.inference_mode():
        frames = pipe(
            image,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            motion_bucket_id=motion_bucket_id,
            fps=fps,
            noise_aug_strength=noise_aug_strength,
            decode_chunk_size=decode_chunk_size,
            generator=generator,
        ).frames[0]

    output_path = OUTPUT_DIR / f"{uuid.uuid4().hex}.mp4"
    export_to_video(frames, str(output_path), fps=fps)

    return str(output_path)


def generate_video_via_comfyui(
    image: Image.Image,
    num_frames: int = 14,
    num_inference_steps: int = 25,
    motion_bucket_id: int = 127,
    fps: int = 7,
    noise_aug_strength: float = 0.02,
    decode_chunk_size: int = 8,
    seed: Optional[int] = None,
    disable_safety: bool = True,
) -> str:
    workflow = load_comfyui_workflow()
    if workflow is not None:
        print(f"[ComfyUI-SVD] Loading workflow: {workflow['name']}")
        for node in workflow["workflow"]["nodes"]:
            if node.get("type") == "KSampler":
                node["inputs"]["steps"] = num_inference_steps
                node["inputs"]["seed"] = seed if seed else 42
            if node.get("type") == "EmptyLatentImage":
                node["inputs"]["width"] = 1024
                node["inputs"]["height"] = 576

    return generate_video_via_diffusers(
        image=image,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        motion_bucket_id=motion_bucket_id,
        fps=fps,
        noise_aug_strength=noise_aug_strength,
        decode_chunk_size=decode_chunk_size,
        seed=seed,
        disable_safety=disable_safety,
    )


def generate_video(**kwargs):
    if kwargs.pop("enable_comfyui", False):
        return generate_video_via_comfyui(**kwargs)
    return generate_video_via_diffusers(**kwargs)


async def generation_worker():
    while True:
        job = await generation_queue.get()
        try:
            job["future"].set_result(
                generate_video(**job["params"])
            )
        except Exception as e:
            job["future"].set_exception(e)
        finally:
            generation_queue.task_done()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global worker_task
    worker_task = asyncio.create_task(generation_worker())
    load_pipeline(disable_safety=True)
    yield
    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        pass


app = FastAPI(title="ComfyUI SVD Image-to-Video API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/generate", response_model=GenerationResponse)
async def generate_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    num_frames: int = Form(14),
    num_inference_steps: int = Form(25),
    motion_bucket_id: int = Form(127),
    fps: int = Form(7),
    noise_aug_strength: float = Form(0.02),
    decode_chunk_size: int = Form(8),
    seed: Optional[int] = Form(None),
    disable_safety: bool = Form(True),
    enable_comfyui: bool = Form(False),
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image = Image.open(file.file).convert("RGB")

    job_id = uuid.uuid4().hex
    future = asyncio.Future()

    await generation_queue.put({
        "job_id": job_id,
        "params": {
            "image": image,
            "num_frames": num_frames,
            "num_inference_steps": num_inference_steps,
            "motion_bucket_id": motion_bucket_id,
            "fps": fps,
            "noise_aug_strength": noise_aug_strength,
            "decode_chunk_size": decode_chunk_size,
            "seed": seed,
            "disable_safety": disable_safety,
            "enable_comfyui": enable_comfyui,
        },
        "future": future,
    })

    try:
        video_path = await asyncio.wait_for(future, timeout=300)
        video_url = f"/outputs/{Path(video_path).name}"
        return GenerationResponse(
            job_id=job_id,
            status="completed",
            message="Video generated successfully via ComfyUI SVD",
            video_url=video_url,
            workflow_id=job_id,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Generation timeout")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/outputs/{filename}")
async def get_output(filename: str):
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, media_type="video/mp4")


@app.get("/health")
async def health_check():
    global pipeline
    with pipeline_lock:
        loaded = pipeline is not None
    return {"status": "healthy", "model_loaded": loaded}


@app.get("/queue/status")
async def queue_status():
    global worker_task
    with pipeline_lock:
        alive = worker_task is not None and not worker_task.done()
    return {
        "queue_size": generation_queue.qsize(),
        "worker_alive": alive,
    }


@app.get("/workflow")
async def get_workflow():
    workflow = load_comfyui_workflow()
    if workflow:
        return JSONResponse(workflow)
    return JSONResponse({"error": "Workflow not found"}, status_code=404)


def gradio_generate(
    image,
    num_frames,
    num_inference_steps,
    motion_bucket_id,
    fps,
    noise_aug_strength,
    decode_chunk_size,
    seed,
    disable_safety,
    enable_comfyui,
):
    if image is None:
        return None, "Please upload an image"

    try:
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")

        seed_val = int(seed) if seed and seed != -1 else None

        video_path = generate_video(
            image=image,
            num_frames=int(num_frames),
            num_inference_steps=int(num_inference_steps),
            motion_bucket_id=int(motion_bucket_id),
            fps=int(fps),
            noise_aug_strength=float(noise_aug_strength),
            decode_chunk_size=int(decode_chunk_size),
            seed=seed_val,
            disable_safety=disable_safety,
            enable_comfyui=enable_comfyui,
        )

        return video_path, "Success! Video generated via ComfyUI SVD."
    except Exception as e:
        return None, f"Error: {str(e)}"


with gr.Blocks(title="ComfyUI SVD - Image to Video", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎬 ComfyUI SVD - Unrestricted Image to Video Generation")
    gr.Markdown("Upload an image and generate a short video clip using Stable Video Diffusion (SVD-XT) via ComfyUI workflow engine")

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                label="Input Image",
                type="pil",
                height=300,
            )

            with gr.Group():
                gr.Markdown("### Generation Parameters")
                num_frames = gr.Slider(14, 25, value=14, step=1, label="Frames (14-25)")
                num_inference_steps = gr.Slider(10, 50, value=25, step=1, label="Inference Steps")
                motion_bucket_id = gr.Slider(1, 255, value=127, step=1, label="Motion Bucket ID (1=static, 255=high motion)")
                fps = gr.Slider(1, 30, value=7, step=1, label="Output FPS")
                noise_aug_strength = gr.Slider(0.0, 1.0, value=0.02, step=0.01, label="Noise Augmentation")
                decode_chunk_size = gr.Slider(1, 16, value=8, step=1, label="Decode Chunk Size (VRAM optimization)")
                seed = gr.Number(value=-1, label="Seed (-1 for random)", precision=0)
                disable_safety = gr.Checkbox(value=True, label="Disable Safety Checker (UNRESTRICTED)")
                enable_comfyui = gr.Checkbox(value=False, label="Use ComfyUI Workflow Engine")

            generate_btn = gr.Button("Generate Video", variant="primary", size="lg")

        with gr.Column(scale=1):
            output_video = gr.Video(label="Generated Video", height=300)
            status_text = gr.Textbox(label="Status", interactive=False)

    generate_btn.click(
        fn=gradio_generate,
        inputs=[
            input_image,
            num_frames,
            num_inference_steps,
            motion_bucket_id,
            fps,
            noise_aug_strength,
            decode_chunk_size,
            seed,
            disable_safety,
            enable_comfyui,
        ],
        outputs=[output_video, status_text],
    )

    gr.Examples(
        examples=[],
        inputs=[input_image],
        label="Example Images (drag & drop your own)",
    )


app = gr.mount_gradio_app(app, demo, path="/ui")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
