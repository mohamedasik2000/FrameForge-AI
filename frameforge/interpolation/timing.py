from typing import Iterator, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)

class TimingScheduler:
    def __init__(self, target_fps: float):
        self.target_fps = target_fps
        self.target_frame_duration = 1.0 / target_fps
        self.target_pts = 0.0
        
        self.prev_img = None
        self.prev_pts = -1.0
        
        self.frame_count_in = 0
        self.frame_count_out = 0

    def push(self, img: Any, pts: float) -> Iterator[Tuple[Any, Any, float]]:
        """
        Push a source frame and its PTS to the scheduler.
        Yields (img0, img1, timestep) for each target frame that falls 
        between the previous frame and this current frame.
        """
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
            logger.warning(f"Non-monotonic PTS detected: prev {self.prev_pts}, current {pts}. Skipping.")
            return

        # Generate all target frames that fall strictly before the current PTS.
        while self.target_pts < pts:
            # Calculate interpolation fraction
            timestep = (self.target_pts - self.prev_pts) / (pts - self.prev_pts)
            # Clip between 0 and 1 just in case of float math quirks
            timestep = max(0.0, min(1.0, timestep))
            
            yield (self.prev_img, img, timestep)
            self.target_pts += self.target_frame_duration
            self.frame_count_out += 1
            
        # Move sliding window
        self.prev_img = img
        self.prev_pts = pts

    def flush(self) -> Iterator[Tuple[Any, Any, float]]:
        """
        Called when the video stream ends to output any final frame.
        """
        # If we have a pending target frame that perfectly matches the last PTS
        if self.prev_img is not None and abs(self.target_pts - self.prev_pts) < 1e-4:
            yield (self.prev_img, self.prev_img, 0.0)
            self.frame_count_out += 1
