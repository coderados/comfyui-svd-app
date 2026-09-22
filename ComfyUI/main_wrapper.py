import os
import sys

# Set up ComfyUI path
COMFYUI_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, COMFYUI_PATH)

os.environ.setdefault("COMFYUI_PATH", COMFYUI_PATH)
os.environ.setdefault("HF_HOME", "./models")
os.environ.setdefault("MODEL_ID", "stabilityai/stable-video-diffusion-img2vid-xt")

if __name__ == "__main__":
    from server import Server
    server = Server()
    server.start()
