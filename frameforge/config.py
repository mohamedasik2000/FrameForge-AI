"""
FrameForge AI configuration.

Configuration is organized into four sections:
- ModelConfig: which model to use, FP16, checksum
- ProcessingConfig: target FPS, scale, GPU
- EncodingConfig: video codec settings
- AudioConfig: audio preservation policy

All fields that the pipeline needs are propagated directly from the config
objects — no intermediate dict with renamed keys.
"""

import os
from typing import Optional

import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    name: str = Field(default="rife-v4.6")
    expected_sha256: Optional[str] = Field(
        default=None,
        description=(
            "SHA-256 of a custom model checkpoint. "
            "Leave None to use the default model with its pinned hash."
        ),
    )
    fp16: bool = Field(
        default=False,
        description=(
            "Use FP16 precision. DISABLED by default because IFNet_m's PReLU "
            "activations can be numerically unstable in FP16 on some GPU architectures. "
            "Enable only after verifying output quality on your hardware."
        ),
    )


class ProcessingConfig(BaseModel):
    target_fps: float = Field(default=60.0, description="Output frame rate.")
    scale: float = Field(
        default=1.0,
        description=(
            "Flow estimation scale. 1.0 = full resolution. "
            "Use 0.5 for 4K to reduce VRAM usage."
        ),
    )
    gpu_id: int = Field(default=0, description="CUDA device index.")


class EncodingConfig(BaseModel):
    codec: str = Field(default="libx264")
    crf: int = Field(default=18)
    preset: str = Field(default="medium")
    pixel_format: str = Field(default="yuv420p")


class AudioConfig(BaseModel):
    preserve: bool = Field(
        default=True,
        description="Preserve audio from source. Set False to produce video-only output.",
    )
    codec: str = Field(
        default="copy",
        description=(
            "Audio codec to use. 'copy' = stream copy (lossless, fastest). "
            "Set to 'aac' to force re-encode. "
            "The encoder will automatically switch to 'aac' for MP4 if the "
            "source codec is not MP4-compatible."
        ),
    )


class FrameForgeConfig(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    encoding: EncodingConfig = Field(default_factory=EncodingConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)


def load_config(path: str) -> FrameForgeConfig:
    if not os.path.exists(path):
        return FrameForgeConfig()
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return FrameForgeConfig.model_validate(data)


def save_config(config: FrameForgeConfig, path: str):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config.model_dump(), f)
