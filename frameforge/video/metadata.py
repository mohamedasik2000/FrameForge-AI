import av
from pydantic import BaseModel
from typing import Optional
import fractions

class VideoMetadata(BaseModel):
    width: int
    height: int
    fps: float
    time_base: float
    duration_secs: float
    frame_count: Optional[int]
    video_codec: str
    has_audio: bool
    audio_codec: Optional[str]
    is_vfr: bool

def inspect_video(path: str) -> VideoMetadata:
    with av.open(path) as container:
        video_stream = next((s for s in container.streams if s.type == 'video'), None)
        if not video_stream:
            raise ValueError("No video stream found in the file.")
            
        audio_stream = next((s for s in container.streams if s.type == 'audio'), None)
        
        # Calculate base properties
        width = video_stream.codec_context.width
        height = video_stream.codec_context.height
        codec = video_stream.codec_context.name
        
        # Framerate handling
        fps_fraction = video_stream.average_rate or video_stream.guessed_rate or fractions.Fraction(24, 1)
        fps = float(fps_fraction)
        time_base = float(video_stream.time_base) if video_stream.time_base else 1/fps
        
        # Duration handling
        duration_secs = 0.0
        if container.duration is not None:
            duration_secs = float(container.duration) / av.time_base
        elif video_stream.duration is not None:
            duration_secs = float(video_stream.duration * video_stream.time_base)
            
        frame_count = video_stream.frames
        if not frame_count or frame_count <= 0:
            frame_count = None # Sometimes av can't read this easily without demuxing the whole file
            
        has_audio = audio_stream is not None
        audio_codec = audio_stream.codec_context.name if has_audio else None
        
        # VFR heuristic: check if average_rate and base_rate differ significantly,
        # or if the stream metadata claims it's VFR. 
        # A more robust check requires scanning PTS, which we do dynamically in the pipeline.
        is_vfr = False
        if video_stream.average_rate and video_stream.base_rate:
            if video_stream.average_rate != video_stream.base_rate:
                is_vfr = True

        return VideoMetadata(
            width=width,
            height=height,
            fps=fps,
            time_base=time_base,
            duration_secs=duration_secs,
            frame_count=frame_count,
            video_codec=codec,
            has_audio=has_audio,
            audio_codec=audio_codec,
            is_vfr=is_vfr
        )
