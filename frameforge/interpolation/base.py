from abc import ABC, abstractmethod
from typing import Any, Optional


class Interpolator(ABC):
    """Abstract base for frame interpolators."""

    @abstractmethod
    def load_model(
        self,
        model_dir: Optional[str] = None,
        model_filename: Optional[str] = None,
    ) -> None:
        """
        Load the model weights.

        Args:
            model_dir: Optional directory containing the model checkpoint.
            model_filename: Optional filename within model_dir.
        """

    @abstractmethod
    def interpolate(self, img0: Any, img1: Any, timestep: float) -> Any:
        """
        Interpolate a frame between img0 and img1.

        Args:
            img0: First frame.
            img1: Second frame.
            timestep: Interpolation position in [0, 1].

        Returns:
            Interpolated frame in the same format as img0/img1.
        """
