"""
Custom ComfyUI nodes for Stable Video Diffusion (SVD) image-to-video generation.
"""
import os
import sys
from typing import Any, Dict, Tuple, List

import torch
import numpy as np
from PIL import Image

# Add ComfyUI to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

class SVDSampler:
    """SVD Sampler node for image-to-video generation"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "latent": ("LATENT",),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFF}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 50}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0}),
                "motion_bucket_id": ("INT", {"default": 127, "min": 1, "max": 255}),
                "fps": ("INT", {"default": 7, "min": 1, "max": 30}),
                "noise_aug_strength": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 1.0}),
                "decode_chunk_size": ("INT", {"default": 8, "min": 1, "max": 16}),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "SVD"
    
    def sample(self, model, latent, seed, steps, cfg, motion_bucket_id, fps, noise_aug_strength, decode_chunk_size):
        # SVD sampling logic
        return (latent,)

class ImageToVideoComfy:
    """Convert image to video using SVD"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": ("MODEL",),
                "num_frames": ("INT", {"default": 14, "min": 14, "max": 25}),
                "motion_bucket_id": ("INT", {"default": 127, "min": 1, "max": 255}),
                "fps": ("INT", {"default": 7, "min": 1, "max": 30}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xFFFFFFFF}),
                "disable_safety": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("VIDEO",)
    FUNCTION = "generate"
    CATEGORY = "SVD"
    
    def generate(self, image, model, num_frames, motion_bucket_id, fps, seed, disable_safety):
        # Image-to-video generation
        frames = torch.randn(num_frames, 576, 1024, 3)  # Placeholder
        return (frames,)

class SVDPipelineLoader:
    """Load SVD pipeline from diffusers"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_id": ("STRING", {"default": "stabilityai/stable-video-diffusion-img2vid-xt"}),
                "torch_dtype": ("STRING", {"default": "float16"}),
                "disable_safety": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("PIPELINE",)
    RETURN_NAMES = ("PIPELINE",)
    FUNCTION = "load"
    CATEGORY = "SVD"
    
    def load(self, model_id, torch_dtype, disable_safety):
        from diffusers import StableVideoDiffusionPipeline
        pipe = StableVideoDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            cache_dir=os.environ.get("HF_HOME", "./models"),
        )
        if disable_safety:
            pipe.safety_checker = None
        return (pipe,)

NODE_CLASS_MAPPINGS = {
    "SVDSampler": SVDSampler,
    "ImageToVideoComfy": ImageToVideoComfy,
    "SVDPipelineLoader": SVDPipelineLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SVDSampler": "SVD Sampler",
    "ImageToVideoComfy": "Image to Video (SVD)",
    "SVDPipelineLoader": "SVD Pipeline Loader",
}
