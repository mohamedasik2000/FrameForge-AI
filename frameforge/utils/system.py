import logging
import signal
import sys
from typing import Callable, List

logger = logging.getLogger(__name__)


class GracefulExit:
    """Register SIGINT/SIGTERM handlers that call cleanup callbacks before exit."""

    def __init__(self):
        self.kill_now = False
        self._callbacks: List[Callable] = []
        signal.signal(signal.SIGINT, self._handle)
        signal.signal(signal.SIGTERM, self._handle)

    def _handle(self, signum, frame):
        logger.warning(f"Received signal {signum}. Initiating graceful shutdown…")
        self.kill_now = True
        for cb in self._callbacks:
            try:
                cb()
            except Exception:
                logger.debug("Exception in shutdown callback", exc_info=True)
        logger.warning("Shutdown complete.")
        sys.exit(1)

    def register_callback(self, callback: Callable):
        self._callbacks.append(callback)


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(levelname)s] %(asctime)s - %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def check_gpu(gpu_id: int = 0) -> None:
    """
    Validate that a usable CUDA device is available at the requested index.

    Raises RuntimeError with actionable messages on any failure.
    No CPU fallback — CUDA is mandatory.
    """
    import torch

    if not isinstance(gpu_id, int):
        raise ValueError(f"gpu_id must be an integer, got {type(gpu_id).__name__!r}: {gpu_id!r}")

    if gpu_id < 0:
        raise RuntimeError(
            f"gpu_id must be >= 0, got {gpu_id}. "
            "Use --gpu 0 for the first GPU, --gpu 1 for the second, etc."
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. An NVIDIA GPU with the CUDA toolkit is required.\n"
            "Check that:\n"
            "  1. An NVIDIA GPU is installed.\n"
            "  2. CUDA drivers are installed (nvidia-smi should work).\n"
            "  3. PyTorch is installed with CUDA support (torch.version.cuda should be set)."
        )

    device_count = torch.cuda.device_count()
    if gpu_id >= device_count:
        raise RuntimeError(
            f"Requested GPU {gpu_id} but only {device_count} GPU(s) are available "
            f"(valid range: 0–{device_count - 1}). Use --gpu 0 to select the first GPU."
        )

    gpu_name = torch.cuda.get_device_name(gpu_id)
    logger.info(f"GPU {gpu_id}: {gpu_name}")
    logger.info(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}")

    # Verify actual allocation (catches driver version mismatches)
    try:
        probe = torch.zeros(1, device=f"cuda:{gpu_id}")
        del probe
    except RuntimeError as e:
        raise RuntimeError(
            f"CUDA device {gpu_id} ({gpu_name}) initialized but failed to allocate memory.\n"
            f"This usually means a driver/CUDA version mismatch.\n"
            f"Original error: {e}"
        ) from e
