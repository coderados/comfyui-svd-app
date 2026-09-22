"""
ComfyUI Manager integration for the SVD Image-to-Video app.
Registers the SVD extension with ComfyUI-Manager.
"""
import os
import json
from pathlib import Path

def register_extension():
    """Register the SVD extension with ComfyUI-Manager"""
    manager_config = Path(__file__).parent / "config.ini"
    
    if manager_config.exists():
        with open(manager_config, "r") as f:
            config = json.load(f)
        
        config["extensions"] = config.get("extensions", {})
        config["extensions"]["ComfyUI-SVD"] = {
            "name": "ComfyUI-SVD",
            "version": "1.0.0",
            "description": "Stable Video Diffusion image-to-video generation",
            "repository": "https://github.com/kijai/ComfyUI-SVD",
            "unrestricted": True,
            "category": "Image-to-Video"
        }
        
        with open(manager_config, "w") as f:
            json.dump(config, f, indent=2)
        print("[ComfyUI-Manager] SVD extension registered")
    else:
        print("[ComfyUI-Manager] Config not found")

if __name__ == "__main__":
    register_extension()
