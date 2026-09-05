from typing import Optional
from pydantic import BaseModel, Field
import yaml
import os

class ModelConfig(BaseModel):
    name: str = Field(default="rife-v4.6")
    expected_sha256: Optional[str] = Field(default=None, description="Expected SHA256 checksum for the model file.")
    fp16: bool = Field(default=True, description="Use FP16 for faster inference if supported.")

class ProcessingConfig(BaseModel):
    target_fps: float = Field(default=60.0)
    scale: float = Field(default=1.0, description="Scale parameter for flow estimation (e.g., 0.5 for 4K).")
    gpu_id: int = Field(default=0)

class EncodingConfig(BaseModel):
    codec: str = Field(default="libx264")
    crf: int = Field(default=18)
    preset: str = Field(default="medium")
    pixel_format: str = Field(default="yuv420p")

class AudioConfig(BaseModel):
    preserve: bool = Field(default=True)
    codec: str = Field(default="copy")

class FrameForgeConfig(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    encoding: EncodingConfig = Field(default_factory=EncodingConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)

def load_config(path: str) -> FrameForgeConfig:
    if not os.path.exists(path):
        return FrameForgeConfig()
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    return FrameForgeConfig.model_validate(data)

def save_config(config: FrameForgeConfig, path: str):
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(config.model_dump(), f)
