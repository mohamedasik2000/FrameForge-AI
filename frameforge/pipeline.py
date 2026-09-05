"""
FrameForge AI pipeline orchestrator.

Responsibilities:
- Validate GPU availability
- Inspect source video metadata
- Initialize TimingScheduler, RIFEInterpolator, VideoStreamReader, FFmpegEncoder
- Drive the main processing loop
- Handle graceful SIGINT/SIGTERM shutdown
- Finalize output atomically after validation

Design notes:
- FFmpegEncoder receives direct references to config objects — no intermediate
  dict with renamed keys.
- TimingScheduler is the single source of truth for frame cadence.
  FFmpegEncoder simply encodes whatever frames are written to it at target_fps.
- Direct-copy frames (timestep == 0.0 or 1.0) bypass RIFE.
"""

import logging
import os
from typing import Dict, Any, Iterator, Optional

from .config import FrameForgeConfig
from .interpolation.rife import RIFEInterpolator
from .interpolation.timing import TimingScheduler
from .utils.system import GracefulExit, check_gpu
from .video.io import FFmpegEncoder, VideoStreamReader
from .video.metadata import VideoMetadata, inspect_video

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, input_path: str, output_path: str, config: FrameForgeConfig):
        self.input_path = input_path
        self.output_path = output_path
        self.config = config

        self.metadata: Optional[VideoMetadata] = None
        self.reader: Optional[VideoStreamReader] = None
        self.encoder: Optional[FFmpegEncoder] = None
        self.scheduler: Optional[TimingScheduler] = None
        self.interpolator: Optional[RIFEInterpolator] = None

        self._frames_written: int = 0

        self.graceful_exit = GracefulExit()
        self.graceful_exit.register_callback(self._cleanup_on_signal)

    # ------------------------------------------------------------------
    # Prepare
    # ------------------------------------------------------------------

    def prepare(self):
        logger.info(f"Preparing pipeline: {self.input_path!r} → {self.output_path!r}")

        # 1. GPU validation (mandatory — no CPU fallback)
        check_gpu(gpu_id=self.config.processing.gpu_id)

        # 2. Input validation and metadata
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"Input file not found: {self.input_path!r}")

        self.metadata = inspect_video(self.input_path)
        logger.info(
            f"Source: {self.metadata.width}x{self.metadata.height} "
            f"@ {self.metadata.fps:.4f} fps, "
            f"{self.metadata.duration_secs:.3f}s, "
            f"codec={self.metadata.video_codec}, "
            f"audio={self.metadata.audio_codec or 'none'}, "
            f"VFR={self.metadata.is_vfr}"
        )
        if self.metadata.is_vfr:
            logger.info("VFR source detected — PTS from stream will be preserved as-is.")

        target_fps = self.config.processing.target_fps
        logger.info(f"Target FPS: {target_fps}")

        # 3. Timing scheduler
        self.scheduler = TimingScheduler(target_fps=target_fps)

        # 4. RIFE interpolator
        logger.info("Loading RIFE interpolator…")
        self.interpolator = RIFEInterpolator(
            fp16=self.config.model.fp16,
            scale=self.config.processing.scale,
            gpu_id=self.config.processing.gpu_id,
            expected_sha256=self.config.model.expected_sha256,
        )
        self.interpolator.load_model()

        # 5. Reader and encoder
        self.reader = VideoStreamReader(self.input_path)

        self.encoder = FFmpegEncoder(
            output_path=self.output_path,
            source_path=self.input_path,
            width=self.metadata.width,
            height=self.metadata.height,
            target_fps=target_fps,
            # AudioConfig fields accessed directly — no renamed dict keys
            audio_preserve=self.config.audio.preserve,
            audio_codec=self.config.audio.codec,
            # EncodingConfig fields
            codec=self.config.encoding.codec,
            preset=self.config.encoding.preset,
            crf=self.config.encoding.crf,
            pixel_format=self.config.encoding.pixel_format,
            has_audio=self.metadata.has_audio,
            source_audio_codec=self.metadata.audio_codec,
        )

    # ------------------------------------------------------------------
    # Process
    # ------------------------------------------------------------------

    def process(self) -> Iterator[Dict[str, Any]]:
        """
        Run the interpolation pipeline.

        Yields progress dicts for the caller (CLI progress bar, API, etc.).
        """
        if self.metadata is None:
            self.prepare()

        logger.info("Starting processing loop…")
        self._frames_written = 0
        self.encoder.start()

        try:
            for img, pts in self.reader.stream_frames():
                if self.graceful_exit.kill_now:
                    logger.warning("Graceful exit requested — stopping processing.")
                    break

                for img0, img1, timestep in self.scheduler.push(img, pts):
                    self._write_output_frame(img0, img1, timestep)

                yield {
                    "frames_in": self.scheduler.frame_count_in,
                    "frames_out": self.scheduler.frame_count_out,
                    "frames_written": self._frames_written,
                    "total_frames": self.metadata.frame_count or 0,
                    "current_pts": float(pts),
                    "duration_secs": self.metadata.duration_secs,
                }

            # Flush: emit any final output frame exactly matching last source PTS
            for img0, img1, timestep in self.scheduler.flush():
                self._write_output_frame(img0, img1, timestep)

        except Exception:
            logger.exception("Pipeline failed during processing")
            raise
        finally:
            self.cleanup()

        logger.info(
            f"Processing complete. "
            f"Input frames: {self.scheduler.frame_count_in}, "
            f"Output frames: {self._frames_written}"
        )

    def _write_output_frame(self, img0, img1, timestep: float):
        """
        Write one scheduled output frame.

        timestep == 0.0 → direct copy img0 (source frame, no RIFE)
        timestep == 1.0 → direct copy img1 (source frame exact match, no RIFE)
        0 < timestep < 1 → RIFE interpolation
        """
        if timestep == 0.0:
            self.encoder.write_frame(img0)
        elif timestep == 1.0:
            self.encoder.write_frame(img1)
        else:
            interpolated = self.interpolator.interpolate(img0, img1, timestep)
            self.encoder.write_frame(interpolated)
        self._frames_written += 1

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self):
        """Close reader and encoder, passing frame count for validation."""
        if self.reader:
            try:
                self.reader.close()
            except Exception:
                logger.debug("Exception closing reader", exc_info=True)
            finally:
                self.reader = None

        if self.encoder:
            try:
                self.encoder.close(frames_written=self._frames_written)
            except Exception:
                logger.exception("Exception closing encoder")
            finally:
                self.encoder = None

    def _cleanup_on_signal(self):
        """Called by GracefulExit on SIGINT/SIGTERM."""
        logger.warning("Signal received — cleaning up pipeline.")
        self.cleanup()
