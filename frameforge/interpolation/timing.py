from typing import Iterator, Tuple, Any, Union
import logging
from fractions import Fraction

logger = logging.getLogger(__name__)

class TimingScheduler:
    def __init__(self, target_fps: Union[int, float, Fraction]):
        self.target_fps = Fraction(target_fps)
        self.target_frame_duration = Fraction(1, self.target_fps)
        self.target_pts = Fraction(0)
        
        self.prev_img = None
        self.prev_pts = Fraction(-1)
        
        self.frame_count_in = 0
        self.frame_count_out = 0

    def push(self, img: Any, pts: Union[float, Fraction]) -> Iterator[Tuple[Any, Any, float]]:
        """
        Push a source frame and its PTS to the scheduler.
        Yields (img0, img1, timestep) for each target frame that falls 
        between the previous frame and this current frame.
        """
        pts = Fraction(pts)
        self.frame_count_in += 1
        
        if self.prev_img is None:
            # First frame
            self.prev_img = img
            self.prev_pts = pts
            self.target_pts = pts # Sync timeline start
            # Yield the first exact frame
            yield (self.prev_img, self.prev_img, 0.0)
            self.target_pts += self.target_frame_duration
            self.frame_count_out += 1
            return

        # Deal with identical PTS (duplicate frames) or backwards PTS
        if pts <= self.prev_pts:
            logger.warning(f"Non-monotonic PTS detected: prev {float(self.prev_pts):.4f}, current {float(pts):.4f}. Skipping.")
            return

        # Generate all target frames that fall strictly before the current PTS.
        # We also want to include the target frame if it exactly matches the current PTS
        # to prevent precision issues, but Fraction math handles this perfectly.
        while self.target_pts < pts:
            # Calculate interpolation fraction
            timestep_frac = (self.target_pts - self.prev_pts) / (pts - self.prev_pts)
            # Convert to float for the model
            timestep = max(0.0, min(1.0, float(timestep_frac)))
            
            yield (self.prev_img, img, timestep)
            self.target_pts += self.target_frame_duration
            self.frame_count_out += 1
            
        self.prev_img = img
        self.prev_pts = pts

    def flush(self) -> Iterator[Tuple[Any, Any, float]]:
        """
        Called when the video stream ends to output any final frame.
        """
        # If we have a pending target frame that perfectly matches the last PTS
        if self.prev_img is not None and self.target_pts == self.prev_pts:
            yield (self.prev_img, self.prev_img, 0.0)
            self.frame_count_out += 1
