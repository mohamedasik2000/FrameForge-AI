import logging
from typing import Dict, Any, Iterator
import os

from .config import FrameForgeConfig
from .utils.system import check_gpu, GracefulExit
from .video.metadata import inspect_video, VideoMetadata
from .video.io import VideoStreamReader, FFmpegEncoder
from .interpolation.timing import TimingScheduler
from .interpolation.rife import RIFEInterpolator

logger = logging.getLogger(__name__)

class Pipeline:
    def __init__(self, input_path: str, output_path: str, config: FrameForgeConfig):
        self.input_path = input_path
        self.output_path = output_path
        self.config = config
        self.metadata: VideoMetadata = None
        
        # Core components
        self.reader = None
        self.encoder = None
        self.scheduler = None
        self.interpolator = None
        
        self.graceful_exit = GracefulExit()
        self.graceful_exit.register_callback(self.cleanup)

    def prepare(self):
        logger.info(f"Preparing pipeline for {self.input_path}")
        
        # 1. System check
        check_gpu(gpu_id=self.config.processing.gpu_id)
        
        # 2. Inspect video
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"Input file not found: {self.input_path}")
            
        self.metadata = inspect_video(self.input_path)
        logger.info(f"Video metadata: {self.metadata.width}x{self.metadata.height} @ {self.metadata.fps}fps, {self.metadata.duration_secs:.2f}s")
        if self.metadata.is_vfr:
            logger.info("Detected possible Variable Frame Rate (VFR). Preserving PTS timeline.")
            
        target_fps = self.config.processing.target_fps
        logger.info(f"Target FPS: {target_fps}")
        
        # 3. Initialize Timing Scheduler
        self.scheduler = TimingScheduler(target_fps=target_fps)
        
        # 4. Initialize RIFE
        logger.info("Initializing RIFE interpolator...")
        self.interpolator = RIFEInterpolator(
            fp16=self.config.model.fp16,
            scale=self.config.processing.scale,
            gpu_id=self.config.processing.gpu_id,
            expected_sha256=self.config.model.expected_sha256
        )
        self.interpolator.load_model() # Will download if missing
        
        # 5. Initialize Reader and Encoder
        self.reader = VideoStreamReader(self.input_path)
        self.encoder = FFmpegEncoder(
            output_path=self.output_path,
            source_path=self.input_path,
            width=self.metadata.width,
            height=self.metadata.height,
            target_fps=target_fps,
            config={
                "codec": self.config.encoding.codec,
                "preset": self.config.encoding.preset,
                "crf": self.config.encoding.crf,
                "pixel_format": self.config.encoding.pixel_format,
                "preserve_audio": self.config.audio.preserve,
                "audio_codec": self.config.audio.codec
            },
            has_audio=self.metadata.has_audio
        )

    def process(self) -> Iterator[Dict[str, Any]]:
        """
        Executes the processing pipeline. 
        Yields progress dictionary which can be consumed by CLI or API.
        """
        if not self.metadata:
            self.prepare()
            
        logger.info("Starting processing loop...")
        self.encoder.start()
        
        try:
            for img, pts in self.reader.stream_frames():
                if self.graceful_exit.kill_now:
                    break
                    
                for img0, img1, t in self.scheduler.push(img, pts):
                    if t == 0.0:
                        self.encoder.write_frame(img0)
                    elif t == 1.0:
                        self.encoder.write_frame(img1)
                    else:
                        out_img = self.interpolator.interpolate(img0, img1, t)
                        self.encoder.write_frame(out_img)
                        
                # Yield progress
                yield {
                    "frames_in": self.scheduler.frame_count_in,
                    "frames_out": self.scheduler.frame_count_out,
                    "total_frames": self.metadata.frame_count or 0,
                    "current_pts": pts,
                    "duration_secs": self.metadata.duration_secs
                }
                
            # Flush remaining frames
            for img0, img1, t in self.scheduler.flush():
                if t == 0.0:
                    self.encoder.write_frame(img0)
                else:
                    out_img = self.interpolator.interpolate(img0, img1, t)
                    self.encoder.write_frame(out_img)
                    
        except Exception as e:
            logger.error(f"Error during processing: {e}")
            raise e
        finally:
            self.cleanup()
            
        logger.info("Processing complete.")

    def cleanup(self):
        logger.debug("Cleaning up pipeline resources...")
        if self.reader:
            self.reader.close()
        if self.encoder:
            self.encoder.close()
