"""
RIFE interpolator for FrameForge AI.

Checkpoint:
    Architecture : ECCV2022-RIFE / Practical-RIFE (hzwer), IFNet_m variant
    File         : flownet.pkl
    Source       : https://huggingface.co/Jacid23/third-eye-models/resolve/main/flownet.pkl
    SHA-256      : 008646e761f0e67cb77f0c6c44cfe3c3e5a05d9d9465311b9681ca650ce030db
    Size         : 21,273,159 bytes (~20 MB)

License: Non-commercial only (follows ECCV2022-RIFE / Practical-RIFE license).

FP16 note:
    IFNet_m uses PReLU activations. PReLU is known to have numerical precision
    issues in FP16 on some GPU architectures. FP16 is therefore DISABLED by
    default. Set config.model.fp16 = True to opt in after verifying stability
    on your hardware.

Scale note:
    scale controls the resolution at which optical flow is estimated.
    scale=1.0  → full resolution (best quality, highest VRAM)
    scale=0.5  → half resolution (lower VRAM, slight quality loss on fast motion)
"""

import os
import hashlib
import logging
import tempfile
import shutil
from fractions import Fraction
from typing import Optional

import numpy as np
import torch
import requests
from tqdm import tqdm

from .base import Interpolator
from .models.rife.RIFE import Model

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default model constants — must be changed together if model changes
# ---------------------------------------------------------------------------

#: Stable filename for the cached checkpoint (independent of download URL)
_DEFAULT_MODEL_FILENAME = "rife_v4.6_flownet.pkl"

#: Download source — publicly accessible PyTorch checkpoint for Practical-RIFE v4.6
#: File format: PyTorch zip state dict (PK header)
#: Architecture: IFNet_m (arbitrary timestep)
_DEFAULT_DOWNLOAD_URL = (
    "https://huggingface.co/Jacid23/third-eye-models/resolve/main/flownet.pkl"
)

#: Pinned SHA-256 for the default checkpoint. Verification is MANDATORY.
#: Computed from the downloaded file on 2026-09-06.
_DEFAULT_SHA256 = "008646e761f0e67cb77f0c6c44cfe3c3e5a05d9d9465311b9681ca650ce030db"

#: HTTP timeout in seconds for model download
_DOWNLOAD_TIMEOUT = 60


class ChecksumError(RuntimeError):
    """Raised when a model file fails SHA-256 verification."""


class RIFEInterpolator(Interpolator):
    """
    RIFE arbitrary-timestep frame interpolator.

    Wraps the ECCV2022-RIFE Model class (IFNet_m variant) and handles:
    - Model weight download and SHA-256 verification
    - CUDA device placement
    - FP16 (opt-in, disabled by default due to PReLU stability)
    - Inference with torch.inference_mode()
    """

    def __init__(
        self,
        fp16: bool = False,
        scale: float = 1.0,
        gpu_id: int = 0,
        expected_sha256: Optional[str] = None,
    ):
        """
        Args:
            fp16: Enable FP16 precision. DISABLED by default.
                  PReLU activations in IFNet_m can be numerically unstable in
                  FP16 on some architectures. Enable only after testing on your GPU.
            scale: Flow estimation scale (1.0 = full res, 0.5 = half res for VRAM saving).
            gpu_id: CUDA device index. Must be a valid device.
            expected_sha256: SHA-256 for a custom model file. If None, the default
                             checkpoint's pinned hash (_DEFAULT_SHA256) is used and
                             verification is still mandatory.
        """
        if not isinstance(gpu_id, int) or gpu_id < 0:
            raise ValueError(f"gpu_id must be a non-negative integer, got {gpu_id!r}")

        self.fp16 = fp16
        self.scale = scale
        self.gpu_id = gpu_id
        self.device = torch.device(f"cuda:{gpu_id}")
        self.model: Optional[Model] = None
        # If caller provides a custom hash, use it; otherwise the default hash is used
        # when loading the default checkpoint.
        self._custom_sha256 = expected_sha256

    # ------------------------------------------------------------------
    # Checksum
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_sha256(filepath: str) -> str:
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for block in iter(lambda: f.read(65536), b""):
                sha256.update(block)
        return sha256.hexdigest().lower()

    def _verify_checksum(self, filepath: str, expected: str) -> None:
        """
        Verify SHA-256 of filepath against expected. Raises ChecksumError on mismatch.

        Verification is always performed — there is no silent skip path.
        """
        logger.info("Verifying model checksum…")
        actual = self._compute_sha256(filepath)
        if actual != expected.lower():
            raise ChecksumError(
                f"Model checksum mismatch!\n"
                f"  File    : {filepath}\n"
                f"  Expected: {expected.lower()}\n"
                f"  Actual  : {actual}\n"
                "Delete the cached file and retry, or verify the download source."
            )
        logger.info(f"Checksum OK: {actual}")

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def _download_weights(self, cache_dir: str, url: str, dest_filename: str) -> str:
        """
        Download weights from url into cache_dir/<dest_filename>.

        - Downloads to a .tmp file
        - Verifies SHA-256 before rename
        - Atomically renames to final path on success
        - Removes .tmp on failure
        """
        os.makedirs(cache_dir, exist_ok=True)
        final_path = os.path.join(cache_dir, dest_filename)
        tmp_path = final_path + ".tmp"

        logger.info(f"Downloading RIFE model from {url} …")
        try:
            resp = requests.get(url, stream=True, timeout=_DOWNLOAD_TIMEOUT)
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))

            with open(tmp_path, "wb") as f:
                for chunk in tqdm(
                    resp.iter_content(chunk_size=65536),
                    total=max(1, total // 65536),
                    unit="chunk",
                    desc="Downloading RIFE",
                    leave=False,
                ):
                    f.write(chunk)

            logger.info("Download complete. Verifying…")
            expected = _DEFAULT_SHA256  # only default URL uses default hash
            self._verify_checksum(tmp_path, expected)

            os.replace(tmp_path, final_path)  # atomic on POSIX; best-effort on Windows
            logger.info(f"Model saved to {final_path}")
            return final_path

        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    # ------------------------------------------------------------------
    # load_model
    # ------------------------------------------------------------------

    def load_model(
        self,
        model_dir: Optional[str] = None,
        model_filename: Optional[str] = None,
    ) -> None:
        """
        Load the RIFE model, downloading if necessary.

        Args:
            model_dir: Directory for the cached checkpoint.
                       Defaults to ~/.cache/frameforge/models/rife-v4.6/
            model_filename: Filename for the cached checkpoint.
                            Defaults to _DEFAULT_MODEL_FILENAME.
        """
        if model_dir is None:
            model_dir = os.path.join(
                os.path.expanduser("~"), ".cache", "frameforge", "models", "rife-v4.6"
            )
        if model_filename is None:
            model_filename = _DEFAULT_MODEL_FILENAME

        weights_path = os.path.join(model_dir, model_filename)

        # Determine which SHA-256 to use
        using_default = self._custom_sha256 is None
        if using_default:
            expected_hash = _DEFAULT_SHA256
        else:
            expected_hash = self._custom_sha256

        if os.path.exists(weights_path):
            logger.info(f"Found cached model at {weights_path}. Verifying…")
            self._verify_checksum(weights_path, expected_hash)
        else:
            if not using_default:
                raise FileNotFoundError(
                    f"Custom model not found: {weights_path}\n"
                    "Place the custom checkpoint at that path. "
                    "Automatic download is only supported for the default model."
                )
            # Download default model
            try:
                weights_path = self._download_weights(
                    model_dir, _DEFAULT_DOWNLOAD_URL, model_filename
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to download RIFE model: {e}\n"
                    f"Manually download the checkpoint from:\n  {_DEFAULT_DOWNLOAD_URL}\n"
                    f"Place it at: {weights_path}\n"
                    f"Expected SHA-256: {_DEFAULT_SHA256}"
                ) from e

        # Initialize model on the correct device
        logger.info(f"Loading RIFE model onto {self.device}…")
        self.model = Model(arbitrary=True, device=self.device)
        self.model.load_model(weights_path, rank=0)
        self.model.eval()

        if self.fp16:
            logger.info(
                "FP16 enabled. Note: PReLU in IFNet_m may be numerically "
                "unstable in FP16 on some GPUs. Disable if you see NaN output."
            )
            self.model.flownet.half()
        else:
            logger.info("FP32 mode (FP16 disabled; set config.model.fp16=True to enable).")

        logger.info("RIFE model loaded successfully.")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def interpolate(self, img0: np.ndarray, img1: np.ndarray, timestep: float) -> np.ndarray:
        """
        Interpolate a frame between img0 and img1.

        Args:
            img0: RGB24 numpy array (H, W, 3), dtype=uint8.
            img1: RGB24 numpy array (H, W, 3), dtype=uint8.
            timestep: Position in [0, 1]. 0.0 → img0, 1.0 → img1.

        Returns:
            RGB24 numpy array (H, W, 3), dtype=uint8.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if not (0.0 <= timestep <= 1.0):
            raise ValueError(f"timestep must be in [0, 1], got {timestep}")

        # (H, W, C) → (1, C, H, W), normalize to [0, 1]
        I0 = (
            torch.from_numpy(img0.transpose(2, 0, 1))
            .unsqueeze(0)
            .to(self.device, non_blocking=True)
            .float()
            .div(255.0)
        )
        I1 = (
            torch.from_numpy(img1.transpose(2, 0, 1))
            .unsqueeze(0)
            .to(self.device, non_blocking=True)
            .float()
            .div(255.0)
        )

        if self.fp16:
            I0 = I0.half()
            I1 = I1.half()

        try:
            mid = self.model.inference(I0, I1, scale=self.scale, timestep=timestep)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                raise RuntimeError(
                    "CUDA out of memory during RIFE inference. "
                    "Try reducing scale (e.g. --scale 0.5) or processing a smaller video."
                ) from e
            raise

        # (1, C, H, W) float → (H, W, C) uint8
        result = (
            mid[0].float().clamp(0.0, 1.0).mul(255.0).byte().cpu().numpy().transpose(1, 2, 0)
        )
        return result
