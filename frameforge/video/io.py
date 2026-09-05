import av
import numpy as np
import subprocess
import logging
import shlex
import os

logger = logging.getLogger(__name__)

class VideoStreamReader:
    def __init__(self, path: str):
        self.path = path
        self.container = None
        self.video_stream = None

    def open(self):
        self.container = av.open(self.path)
        self.video_stream = next((s for s in self.container.streams if s.type == 'video'), None)
        if not self.video_stream:
            raise ValueError("No video stream found.")
        self.video_stream.thread_type = "AUTO" # Enable multithreaded decoding

    def stream_frames(self):
        """Yields (frame_rgb24_numpy, pts_seconds)"""
        if not self.container:
            self.open()
            
        time_base = float(self.video_stream.time_base)
        for frame in self.container.decode(video=0):
            pts_seconds = frame.pts * time_base if frame.pts is not None else 0.0
            # Convert to numpy array in RGB24 format
            img = frame.to_ndarray(format='rgb24')
            yield img, pts_seconds

    def close(self):
        if self.container:
            self.container.close()

class FFmpegEncoder:
    """
    Spawns an FFmpeg subprocess that accepts raw RGB24 frames via stdin
    and muxes them with the original audio directly from the source file.
    """
    def __init__(self, 
                 output_path: str, 
                 source_path: str, 
                 width: int, 
                 height: int, 
                 target_fps: float,
                 config: dict,
                 has_audio: bool):
        self.output_path = output_path
        self.source_path = source_path
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self.config = config
        self.has_audio = has_audio
        self.process = None

    def start(self):
        # We output to a temporary .part file to ensure atomic finalization
        self.part_path = self.output_path + ".part"
        if os.path.exists(self.part_path):
            os.remove(self.part_path)
            
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{self.width}x{self.height}",
            "-pix_fmt", "rgb24",
            "-framerate", str(self.target_fps), # strict CFR input frame rate
            "-i", "-", # Input 0: raw frames from stdin
        ]

        if self.has_audio and self.config.get("preserve_audio", True):
            cmd.extend(["-i", self.source_path]) # Input 1: original video for audio
            
        # Video encoding settings
        cmd.extend([
            "-c:v", self.config.get("codec", "libx264"),
            "-preset", self.config.get("preset", "medium"),
            "-crf", str(self.config.get("crf", 18)),
            "-pix_fmt", self.config.get("pixel_format", "yuv420p"),
            "-r", str(self.target_fps) # Ensure strict CFR output
        ])
        
        # Audio mapping and encoding settings
        if self.has_audio and self.config.get("preserve_audio", True):
            # If output is mp4, copy might fail if original is an unsupported codec like FLAC or PCM
            # For robustness, we will use aac by default unless specified
            audio_codec = self.config.get("audio_codec", "aac" if self.output_path.lower().endswith(".mp4") else "copy")
            cmd.extend([
                "-map", "0:v",   # Take video from input 0 (stdin)
                "-map", "1:a?",  # Take audio from input 1 (original video)
                "-c:a", audio_codec,
                "-shortest"      # End encoding when the shortest stream ends
            ])
        else:
            cmd.extend(["-map", "0:v"]) # Take video from input 0 only

        cmd.append(self.part_path)
        
        cmd_str = " ".join(shlex.quote(arg) for arg in cmd)
        logger.debug(f"Starting encoder: {cmd_str}")

        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, # Suppress ffmpeg output to keep console clean
            stderr=subprocess.PIPE
        )

    def write_frame(self, frame_rgb24: np.ndarray):
        if not self.process:
            raise RuntimeError("Encoder not started.")
        try:
            self.process.stdin.write(frame_rgb24.tobytes())
        except BrokenPipeError:
            # Handle early termination or ffmpeg crash
            _, stderr = self.process.communicate()
            raise RuntimeError(f"FFmpeg encoder crashed: {stderr.decode()}")

    def _validate_output(self):
        """Run ffprobe on the output part file to ensure it's a valid video."""
        try:
            cmd = [
                "ffprobe", 
                "-v", "error", 
                "-select_streams", "v:0", 
                "-show_entries", "stream=duration,nb_frames,r_frame_rate", 
                "-of", "default=noprint_wrappers=1:nokey=1", 
                self.part_path
            ]
            output = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
            lines = output.strip().split("\n")
            if not lines or len(lines) < 2:
                logger.warning("ffprobe could not determine output validity.")
                return
            logger.info(f"Output validation passed: {output.strip().replace(chr(10), ' | ')}")
        except subprocess.CalledProcessError as e:
            logger.error(f"FFprobe validation failed: {e.output}")
            raise RuntimeError(f"Output video {self.part_path} failed validation and may be corrupt.")
        except FileNotFoundError:
            logger.warning("ffprobe not found. Skipping output validation.")

    def close(self):
        if self.process:
            self.process.stdin.close()
            self.process.wait()
            if self.process.returncode != 0:
                stderr = self.process.stderr.read().decode()
                raise RuntimeError(f"FFmpeg encoder failed with code {self.process.returncode}. stderr: {stderr}")
            else:
                # Validate before atomic rename
                self._validate_output()
                if os.path.exists(self.output_path):
                    os.remove(self.output_path)
                os.rename(self.part_path, self.output_path)
