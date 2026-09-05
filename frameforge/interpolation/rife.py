import os
import torch
import numpy as np
import logging
import requests
import hashlib
from tqdm import tqdm
import shutil

from .base import Interpolator
from .models.rife.RIFE import Model

logger = logging.getLogger(__name__)

# Fallback verified URL for v4.6 weights (if available on huggingface or github)
DEFAULT_WEIGHTS_URL = "https://github.com/styler00dollar/VapourSynth-RIFE-ncnn-Vulkan/releases/download/models/rife46.pth"

class RIFEInterpolator(Interpolator):
    def __init__(self, fp16: bool = True, scale: float = 1.0, gpu_id: int = 0, expected_sha256: str = None):
        self.fp16 = fp16
        self.scale = scale
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        if self.device.type != 'cuda':
            raise RuntimeError("CUDA is required for RIFE interpolation, but no CUDA device was found.")
        self.model = None
        self.expected_sha256 = expected_sha256

    def _verify_checksum(self, filepath: str, expected_hash: str) -> bool:
        if not expected_hash:
            logger.warning("No expected SHA-256 checksum provided; skipping verification.")
            return True
        logger.info("Verifying model checksum...")
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        actual_hash = sha256_hash.hexdigest()
        logger.info(f"Actual Hash:   {actual_hash}")
        logger.info(f"Expected Hash: {expected_hash}")
        return actual_hash.lower() == expected_hash.lower()

    def _download_weights(self, cache_dir: str, url: str) -> str:
        os.makedirs(cache_dir, exist_ok=True)
        # Using rife46.pth or flownet.pkl depending on the URL
        filename = url.split('/')[-1]
        weights_path = os.path.join(cache_dir, filename)
        
        if os.path.exists(weights_path):
            if self._verify_checksum(weights_path, self.expected_sha256):
                logger.info(f"Verified model weights found at {weights_path}")
                return weights_path
            else:
                logger.warning(f"Existing model weights at {weights_path} failed checksum validation. Redownloading...")
                os.remove(weights_path)

        tmp_path = weights_path + ".tmp"
        logger.info(f"Downloading model weights from {url}...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024
            
            with open(tmp_path, 'wb') as f:
                for data in tqdm(response.iter_content(block_size), total=total_size//block_size, unit='KB'):
                    f.write(data)
            
            # Verify checksum on the temp file before renaming
            if not self._verify_checksum(tmp_path, self.expected_sha256):
                raise ValueError("Downloaded model failed SHA-256 checksum verification!")
                
            os.rename(tmp_path, weights_path)
            logger.info("Download and verification complete.")
            return weights_path
        except Exception as e:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise e

    def load_model(self, model_dir: str = None, model_filename: str = None):
        if model_dir is None:
            # Default to local cache
            model_dir = os.path.join(os.path.expanduser("~"), ".cache", "frameforge", "models", "rife-v4.6")
            
        if model_filename is None:
            model_filename = "rife46.pth"
            
        weights_path = os.path.join(model_dir, model_filename)
        
        if not os.path.exists(weights_path):
            try:
                weights_path = self._download_weights(model_dir, DEFAULT_WEIGHTS_URL)
            except Exception as e:
                logger.error(f"Failed to automatically download weights: {e}")
                logger.error(f"Please place '{model_filename}' in {model_dir} manually.")
                raise e
        else:
            if not self._verify_checksum(weights_path, self.expected_sha256):
                raise ValueError("Cached model failed SHA-256 checksum verification. Delete the file to redownload.")

        # Initialize the model using the embedded RIFE code
        self.model = Model(arbitrary=True) # use IFNet_m for arbitrary timestep
        self.model.load_model(weights_path, -1)
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
            mid = (mid[0] * 255.).clamp(0, 255).byte().cpu().numpy().transpose(1, 2, 0)
            return mid
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error("CUDA Out of Memory! Try reducing the processing scale (e.g. --scale 0.5) or using a smaller video.")
            raise e
