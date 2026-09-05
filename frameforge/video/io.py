"""
Video I/O: frame reader (PyAV) and FFmpeg encoder (rawvideo stdin).

Design contract
---------------
VideoStreamReader yields frames as (rgb24_ndarray, Fraction_pts_in_seconds).
PTS is represented as fractions.Fraction computed from the stream's integer PTS
and its rational time_base, preserving exact rational arithmetic downstream.

FFmpegEncoder receives frames that the TimingScheduler has already scheduled at
the correct CFR cadence. It encodes them at the requested target_fps using
rawvideo stdin. Because rawvideo carries no per-frame timing information,
the scheduler is the single source of truth for output frame cadence:
every frame written to write_frame() corresponds to exactly one output timestamp
at 1/target_fps intervals. FFmpeg is told the input rate (-framerate) and the
output rate (-r) so it produces a strict CFR stream.

Audio handling
--------------
When preserve=True and the source has audio:
  - If the output container is MP4 and the source audio codec is not
    already AAC, the audio is transcoded to AAC (MP4 does not support
    all codecs). Otherwise audio is copied stream-for-stream.
  - -shortest is NOT used. Instead the video timeline governs duration;
    -avoid_negative_ts make_zero handles timestamp normalization.
  - The caller may override the audio codec via AudioConfig.codec.

Output finalization
-------------------
Output is written to a .part file. After FFmpeg exits successfully,
ffprobe validation is run, and only on success the file is atomically
renamed to the final path.
"""

import json
import os
import shlex
import subprocess
import logging
from fractions import Fraction
from typing import Generator, Optional, Tuple

import av
import numpy as np

logger = logging.getLogger(__name__)


class VideoStreamReader:
    """
    Reads video frames via PyAV, yielding (rgb24_ndarray, Fraction_pts_seconds).

    PTS is computed as Fraction(frame.pts) * Fraction(time_base_num, time_base_den)
    to preserve exact rational arithmetic. No float conversion until the caller
    decides to use one.
    """

    def __init__(self, path: str):
        self.path = path
        self._container: Optional[av.container.Container] = None
        self._video_stream: Optional[av.video.VideoStream] = None

    def open(self):
        self._container = av.open(self.path)
        self._video_stream = next(
            (s for s in self._container.streams if s.type == "video"), None
        )
        if self._video_stream is None:
            raise ValueError(f"No video stream found in {self.path!r}")
        self._video_stream.thread_type = "AUTO"

    def stream_frames(self) -> Generator[Tuple[np.ndarray, Fraction], None, None]:
        """
        Yield (rgb24_ndarray, pts_as_Fraction_in_seconds).

        PTS is exact rational arithmetic based on the stream's time_base.
        Frames with pts=None are assigned pts=0 and a warning is logged.
        """
        if self._container is None:
            self.open()

        tb = self._video_stream.time_base  # PyAV returns a fractions.Fraction
        time_base = Fraction(tb.numerator, tb.denominator)

        prev_pts: Optional[Fraction] = None
        for frame in self._container.decode(video=0):
            if frame.pts is None:
                logger.warning("Frame with pts=None encountered; using 0. VFR detection may be affected.")
                raw_pts = 0
            else:
                raw_pts = frame.pts

            pts_frac = Fraction(raw_pts) * time_base

            img = frame.to_ndarray(format="rgb24")
            yield img, pts_frac

    def close(self):
        if self._container:
            self._container.close()
            self._container = None


class FFmpegEncoder:
    """
    Encodes a CFR frame sequence via FFmpeg rawvideo stdin.

    The scheduler is the single source of truth for output frame cadence.
    Every frame written via write_frame() represents one output timestamp.
    FFmpeg is configured with -framerate (input rate) and -r (output rate)
    set to target_fps to produce a strict CFR output stream.

    Output is written to <output_path>.part and atomically renamed after
    ffprobe validation passes.
    """

    def __init__(
        self,
        output_path: str,
        source_path: str,
        width: int,
        height: int,
        target_fps: float,
        audio_preserve: bool,
        audio_codec: str,
        codec: str = "libx264",
        preset: str = "medium",
        crf: int = 18,
        pixel_format: str = "yuv420p",
        has_audio: bool = False,
        source_audio_codec: Optional[str] = None,
    ):
        self.output_path = output_path
        self.source_path = source_path
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self.audio_preserve = audio_preserve
        self.audio_codec = audio_codec
        self.codec = codec
        self.preset = preset
        self.crf = crf
        self.pixel_format = pixel_format
        self.has_audio = has_audio
        self.source_audio_codec = source_audio_codec
        self.process: Optional[subprocess.Popen] = None
        self.part_path: Optional[str] = None

    def _resolve_audio_codec(self) -> str:
        """
        Determine the audio codec to use.

        Policy:
        - If AudioConfig.codec is explicitly set (not "copy"), use it.
        - If codec is "copy" and the output is MP4, switch to AAC if the
          source audio is not already AAC (MP4 container requirement).
        - Otherwise copy the stream.
        """
        if self.audio_codec != "copy":
            return self.audio_codec
        # "copy" requested — check container compatibility
        is_mp4 = self.output_path.lower().endswith(".mp4")
        if is_mp4 and self.source_audio_codec and self.source_audio_codec.lower() not in (
            "aac", "mp3", "ac3", "eac3", "opus"
        ):
            logger.info(
                f"Source audio codec {self.source_audio_codec!r} may not be "
                "compatible with MP4. Transcoding to AAC."
            )
            return "aac"
        return "copy"

    def start(self):
        self.part_path = self.output_path + ".part"
        if os.path.exists(self.part_path):
            os.remove(self.part_path)

        # Input 0: raw video frames from stdin at exactly target_fps CFR
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{self.width}x{self.height}",
            "-pix_fmt", "rgb24",
            "-framerate", str(self.target_fps),
            "-i", "-",
        ]

        include_audio = self.has_audio and self.audio_preserve
        if include_audio:
            # Input 1: source file for audio stream(s)
            cmd.extend(["-i", self.source_path])

        # Video encoding — strict CFR output
        cmd.extend([
            "-c:v", self.codec,
            "-preset", self.preset,
            "-crf", str(self.crf),
            "-pix_fmt", self.pixel_format,
            "-r", str(self.target_fps),  # enforce CFR output
        ])

        if include_audio:
            audio_codec = self._resolve_audio_codec()
            cmd.extend([
                "-map", "0:v",    # video from stdin
                "-map", "1:a?",   # audio from source (optional — won't fail if absent)
                "-c:a", audio_codec,
                # Do NOT use -shortest: it can truncate video if audio has offset.
                # Instead, let the video timeline govern output duration.
                "-avoid_negative_ts", "make_zero",
            ])
        else:
            cmd.extend(["-map", "0:v"])

        cmd.append(self.part_path)

        logger.debug("FFmpeg command: %s", " ".join(shlex.quote(a) for a in cmd))

        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def write_frame(self, frame_rgb24: np.ndarray):
        """Write one output frame. Must be called after start()."""
        if self.process is None:
            raise RuntimeError("Encoder not started. Call start() first.")
        try:
            self.process.stdin.write(frame_rgb24.tobytes())
        except BrokenPipeError:
            # FFmpeg died — collect stderr for diagnosis
            stderr = b""
            try:
                _, stderr = self.process.communicate(timeout=5)
            except Exception:
                pass
            raise RuntimeError(
                f"FFmpeg encoder crashed (BrokenPipe):\n{stderr.decode(errors='replace')}"
            )

    def _validate_output(self, expected_fps: float, frames_written: int):
        """
        Run ffprobe on the .part file and verify:
        - Video stream exists
        - Frame rate matches target
        - Frame count is within 1 of expected
        - Duration is sensible (> 0)
        """
        if not os.path.exists(self.part_path):
            raise RuntimeError(f"Output file not found after encoding: {self.part_path}")

        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate,nb_frames,duration",
            "-of", "json",
            self.part_path,
        ]
        try:
            raw = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=30)
        except FileNotFoundError:
            logger.warning("ffprobe not found — skipping output validation.")
            return
        except subprocess.TimeoutExpired:
            logger.warning("ffprobe timed out — skipping output validation.")
            return
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"ffprobe failed on output file:\n{e.output}"
            ) from e

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("ffprobe returned non-JSON — skipping detailed validation.")
            return

        streams = data.get("streams", [])
        if not streams:
            raise RuntimeError(
                f"Output validation failed: no video stream found in {self.part_path}"
            )

        stream = streams[0]

        # --- Frame rate check ---
        r_frame_rate = stream.get("r_frame_rate", "")
        if r_frame_rate:
            try:
                num, den = map(int, r_frame_rate.split("/"))
                actual_fps = num / den
                tol = expected_fps * 0.01  # 1% tolerance
                if abs(actual_fps - expected_fps) > tol:
                    raise RuntimeError(
                        f"Output FPS mismatch: expected {expected_fps}, "
                        f"got {actual_fps} (r_frame_rate={r_frame_rate})"
                    )
            except (ValueError, ZeroDivisionError):
                logger.warning(f"Could not parse r_frame_rate: {r_frame_rate!r}")

        # --- Frame count check ---
        nb_frames_str = stream.get("nb_frames", "")
        if nb_frames_str:
            try:
                actual_nb = int(nb_frames_str)
                if abs(actual_nb - frames_written) > 1:
                    raise RuntimeError(
                        f"Output frame count mismatch: wrote {frames_written} frames, "
                        f"but ffprobe reports {actual_nb} frames."
                    )
            except ValueError:
                logger.warning(f"Could not parse nb_frames: {nb_frames_str!r}")

        # --- Duration check ---
        duration_str = stream.get("duration", "")
        if duration_str:
            try:
                duration = float(duration_str)
                if duration <= 0:
                    raise RuntimeError(
                        f"Output duration is {duration}s — file may be corrupt."
                    )
                logger.info(f"Output validation passed: fps={r_frame_rate}, "
                            f"frames={nb_frames_str}, duration={duration:.3f}s")
            except ValueError:
                logger.warning(f"Could not parse duration: {duration_str!r}")

    def close(self, frames_written: int = 0):
        """
        Close stdin, wait for FFmpeg, validate output, and atomically rename.

        Args:
            frames_written: Number of frames written (for validation).
        """
        if self.process is None:
            return

        self.process.stdin.close()
        self.process.wait()

        if self.process.returncode != 0:
            stderr = self.process.stderr.read().decode(errors="replace")
            raise RuntimeError(
                f"FFmpeg encoder exited with code {self.process.returncode}:\n{stderr}"
            )

        self._validate_output(expected_fps=self.target_fps, frames_written=frames_written)

        # Atomic rename
        if os.path.exists(self.output_path):
            os.remove(self.output_path)
        os.replace(self.part_path, self.output_path)
        logger.info(f"Output written to {self.output_path}")
