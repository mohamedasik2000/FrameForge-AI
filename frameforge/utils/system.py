import signal
import sys
import logging
from typing import Callable, List

logger = logging.getLogger(__name__)

class GracefulExit:
    def __init__(self):
        self.kill_now = False
        self.callbacks: List[Callable] = []
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        logger.warning(f"Received signal {signum}. Initiating graceful shutdown...")
        self.kill_now = True
        for callback in self.callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Error during shutdown callback: {e}")
        logger.warning("Shutdown complete. Exiting.")
        sys.exit(1)

    def register_callback(self, callback: Callable):
        self.callbacks.append(callback)

def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(levelname)s] %(asctime)s - %(name)s - %(message)s",
        datefmt="%H:%M:%S"
    )

def check_gpu():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. An NVIDIA GPU is required for this application.")
    
    device_count = torch.cuda.device_count()
    gpu_name = torch.cuda.get_device_name(0)
    logger.info(f"Detected {device_count} GPU(s). Primary GPU: {gpu_name}")
    
    # Check PyTorch version and CUDA version
    logger.info(f"PyTorch Version: {torch.__version__}")
    logger.info(f"CUDA Version (compiled with PyTorch): {torch.version.cuda}")
    
    return True
