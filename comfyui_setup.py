import os
import sys
import json
from pathlib import Path

COMFYUI_PATH = Path(__file__).parent / "ComfyUI"
WORKFLOW_PATH = Path(__file__).parent / "workflows" / "svd_image_to_video.json"

def install_comfyui_svd():
    """Install ComfyUI-SVD custom node"""
    custom_nodes = COMFYUI_PATH / "custom_nodes"
    svd_path = custom_nodes / "ComfyUI-SVD"
    if not svd_path.exists():
        import subprocess
        subprocess.run([
            "git", "clone", "--depth", "1",
            "https://github.com/kijai/ComfyUI-SVD.git",
            str(svd_path)
        ], cwd=str(custom_nodes))
        print("[ComfyUI-SVD] Installed successfully")
    else:
        print("[ComfyUI-SVD] Already installed")

def setup_workflow():
    """Setup ComfyUI workflow JSON"""
    if not WORKFLOW_PATH.exists():
        print("[ComfyUI] Workflow not found, creating default...")
        default_workflow = {
            "name": "svd_image_to_video",
            "version": "1.0.0",
            "workflow": {"nodes": [], "links": []},
            "parameters": {"model": "stabilityai/stable-video-diffusion-img2vid-xt"}
        }
        WORKFLOW_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(WORKFLOW_PATH, "w") as f:
            json.dump(default_workflow, f, indent=2)
        print("[ComfyUI] Default workflow created")

def launch_comfyui():
    """Launch ComfyUI server with SVD nodes"""
    os.environ["COMFYUI_PATH"] = str(COMFYUI_PATH)
    os.chdir(COMFYUI_PATH)
    sys.argv = ["python", "main.py", "--listen", "0.0.0.0", "--port", "8188"]
    import main
    main.main()

if __name__ == "__main__":
    install_comfyui_svd()
    setup_workflow()
    launch_comfyui()
