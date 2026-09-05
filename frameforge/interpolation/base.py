from abc import ABC, abstractmethod
from typing import Any

class Interpolator(ABC):
    @abstractmethod
    def load_model(self, model_dir: str):
        pass

    @abstractmethod
    def interpolate(self, img0: Any, img1: Any, timestep: float) -> Any:
        """
        Interpolate a frame between img0 and img1 at the given timestep [0, 1].
        """
        pass
