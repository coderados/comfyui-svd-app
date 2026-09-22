"""
ComfyUI workflow definitions for SVD image-to-video generation.
"""
import json
from pathlib import Path

WORKFLOW_PATH = Path(__file__).parent / "svd_image_to_video.json"
NODES_PATH = Path(__file__).parent / "svd_nodes.json"

def load_workflow() -> dict:
    """Load the SVD ComfyUI workflow JSON"""
    if WORKFLOW_PATH.exists():
        with open(WORKFLOW_PATH) as f:
            return json.load(f)
    return None

def load_nodes() -> dict:
    """Load the SVD custom node definitions"""
    if NODES_PATH.exists():
        with open(NODES_PATH) as f:
            return json.load(f)
    return None

SVD_WORKFLOW = load_workflow()
SVD_NODES = load_nodes()

__all__ = ["load_workflow", "load_nodes", "SVD_WORKFLOW", "SVD_NODES"]
