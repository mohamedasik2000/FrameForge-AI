import os
import torch
import numpy as np
import logging
from typing import Any
import requests
import zipfile
import hashlib
from tqdm import tqdm

from .base import Interpolator

# Adjust import path for the embedded RIFE model
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "models", "rife"))
from RIFE import Model

logger = logging.getLogger(__name__)

# Fallback verified URL for v4.6 weights (if available on huggingface or github)
# Note: For production, this should point to a stable mirror.
DEFAULT_WEIGHTS_URL = "https://github.com/styler00dollar/VapourSynth-RIFE-ncnn-Vulkan/releases/download/models/rife46.pth"

class RIFEInterpolator(Interpolator):
    def __init__(self, fp16: bool = True, scale: float = 1.0, gpu_id: int = 0):
        self.fp16 = fp16
        self.scale = scale
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        self.model = None

    def _download_weights(self, cache_dir: str, url: str) -> str:
        os.makedirs(cache_dir, exist_ok=True)
        # Using a fixed name for the downloaded weights
        weights_path = os.path.join(cache_dir, "flownet.pkl")
        
        if os.path.exists(weights_path):
            logger.info(f"Model weights found at {weights_path}")
            return weights_path

        logger.info(f"Downloading model weights from {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024
        
        with open(weights_path, 'wb') as f:
            for data in tqdm(response.iter_content(block_size), total=total_size//block_size, unit='KB'):
                f.write(data)
                
        logger.info("Download complete.")
        return weights_path

    def load_model(self, model_dir: str = None):
        if model_dir is None:
            # Default to local cache
            model_dir = os.path.join(os.path.dirname(__file__), "..", "..", "models", "rife-v4.6")
            
        weights_path = os.path.join(model_dir, "flownet.pkl")
        if not os.path.exists(weights_path):
            # Attempt to download
            try:
                self._download_weights(model_dir, DEFAULT_WEIGHTS_URL)
            except Exception as e:
                logger.error(f"Failed to automatically download weights: {e}")
                logger.error("Please place 'flownet.pkl' in the models directory manually.")
                raise e

        # Initialize the model using the embedded RIFE code
        self.model = Model()
        self.model.load_model(model_dir, -1)
        self.model.eval()
        self.model.device()
        
        if self.fp16 and self.device.type == 'cuda':
            logger.info("Enabling FP16 inference for RIFE.")
            self.model.flownet.half()

    @torch.inference_mode()
    def interpolate(self, img0: np.ndarray, img1: np.ndarray, timestep: float) -> np.ndarray:
        """
        img0, img1: RGB24 numpy arrays (H, W, 3)
        Returns: RGB24 numpy array
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
            
        # Convert to torch tensor (B, C, H, W), normalize to [0, 1]
        I0 = torch.from_numpy(np.transpose(img0, (2, 0, 1))).unsqueeze(0).to(self.device, non_blocking=True).float() / 255.
        I1 = torch.from_numpy(np.transpose(img1, (2, 0, 1))).unsqueeze(0).to(self.device, non_blocking=True).float() / 255.
        
        if self.fp16 and self.device.type == 'cuda':
            I0 = I0.half()
            I1 = I1.half()
            
        try:
            # RIFE model inference
            mid = self.model.inference(I0, I1, scale=self.scale, timestep=timestep)
            
            # Convert back to numpy (H, W, C)
            mid = (mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)
            return mid
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error("CUDA Out of Memory! Try reducing the processing scale (e.g. --scale 0.5) or using a smaller video.")
            raise e
