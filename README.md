# FrameForge AI

A production-ready AI video frame interpolation application using Practical-RIFE.

## Features
- AI Frame Interpolation (e.g. 24 -> 60 FPS)
- Variable Frame Rate (VFR) aware streaming processing
- Clean, decoupled pipeline designed for extensibility
- Automatic model downloading and verification
- Preserves original audio synchronously

## Requirements
- Python 3.11+
- NVIDIA GPU with CUDA
- FFmpeg installed and in PATH
- PyTorch

## Installation

```bash
git clone https://github.com/your-org/frameforge-ai.git
cd frameforge-ai

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## Usage

```bash
python -m frameforge --input input.mp4 --output output.mp4 --fps 60
```
