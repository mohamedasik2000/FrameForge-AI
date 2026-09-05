# FrameForge AI

GPU-accelerated AI video frame interpolation using Practical-RIFE (ECCV2022).
Converts 24 FPS → 60 FPS (or any target frame rate) with full VFR support and
audio preservation.

[![GitHub](https://img.shields.io/badge/GitHub-mohamedasik2000%2FFrameForge--AI-blue)](https://github.com/mohamedasik2000/FrameForge-AI)

---

## Features

- **AI Frame Interpolation** — 24→60 FPS, 30→60 FPS, and any arbitrary conversion
- **VFR-aware** — preserves source PTS; never forces VFR→CFR for decoding
- **Exact rational timing** — uses `fractions.Fraction` arithmetic throughout
- **Audio preserved** — original audio is stream-copied or re-encoded for container compatibility
- **Automatic model download** — downloads and SHA-256 verifies the RIFE checkpoint on first run
- **Atomic output** — writes to `.part` file, validates with ffprobe, then renames atomically
- **CUDA mandatory** — no silent CPU fallback; fails fast with a clear error

---

## Requirements

| Requirement | Version |
|---|---|
| Python | 3.11+ |
| NVIDIA GPU | CUDA 11.8+ |
| PyTorch | 2.0+ with CUDA |
| FFmpeg | 4.4+ (in PATH) |
| ffprobe | same package as FFmpeg |

---

## Installation

```bash
git clone https://github.com/mohamedasik2000/FrameForge-AI.git
cd FrameForge-AI/frameforge-ai

# Create a virtual environment
python -m venv .venv

# Activate (Linux/macOS)
source .venv/bin/activate
# Activate (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

---

## Model Download and Cache

On first run, FrameForge AI will automatically download the RIFE model:

| Property | Value |
|---|---|
| Architecture | ECCV2022-RIFE / Practical-RIFE (hzwer), IFNet_m variant |
| Checkpoint | `flownet.pkl` |
| Size | ~20 MB |
| Cache location | `~/.cache/frameforge/models/rife-v4.6/rife_v4.6_flownet.pkl` |
| SHA-256 | `008646e761f0e67cb77f0c6c44cfe3c3e5a05d9d9465311b9681ca650ce030db` |

**Verification is mandatory** — the SHA-256 is always checked before loading.
The model will not load if the checksum does not match.

### Manual installation (offline)

Download from:
```
https://huggingface.co/Jacid23/third-eye-models/resolve/main/flownet.pkl
```

Place at:
```
~/.cache/frameforge/models/rife-v4.6/rife_v4.6_flownet.pkl
```

---

## CLI Usage

```bash
# 24 FPS → 60 FPS (default)
python -m frameforge --input input.mp4 --output output.mp4

# Custom target FPS
python -m frameforge --input input.mp4 --output output.mp4 --fps 48

# Scale down flow estimation for VRAM savings (e.g. 4K video)
python -m frameforge --input 4k.mp4 --output 4k_60fps.mp4 --scale 0.5

# Select GPU by index
python -m frameforge --input input.mp4 --output output.mp4 --gpu 1

# Use a config file
python -m frameforge --input input.mp4 --output output.mp4 --config config.yaml

# Verbose logging
python -m frameforge --input input.mp4 --output output.mp4 --verbose
```

---

## Configuration

Create a `config.yaml` to override defaults:

```yaml
model:
  fp16: false         # FP16 disabled by default (see FP16 section below)
  expected_sha256: null  # null = use pinned default hash

processing:
  target_fps: 60.0
  scale: 1.0          # 1.0 = full resolution, 0.5 = half (saves VRAM)
  gpu_id: 0

encoding:
  codec: libx264
  crf: 18
  preset: medium
  pixel_format: yuv420p

audio:
  preserve: true
  codec: copy         # 'copy' = stream copy; 'aac' = re-encode
```

---

## VFR → CFR Behavior

FrameForge AI accepts Variable Frame Rate (VFR) input without forcing it to CFR
for decoding. Source PTS values from the stream are preserved as exact rational
numbers throughout the pipeline.

Output is always **Constant Frame Rate (CFR)** at the requested `--fps`.

For each output frame at timestamp `t_n = source_start + n / target_fps`:
- If `t_n` exactly matches a source frame: source frame is emitted directly (no AI)
- If `t_n` falls between two source frames: RIFE interpolates with timestep
  `= (t_n - prev_pts) / (next_pts - prev_pts)`

---

## Audio Behavior

| Scenario | Behavior |
|---|---|
| Source has audio, `audio.preserve: true` | Audio stream copied from source |
| Source is MP4 with non-AAC audio | Transcoded to AAC automatically |
| Source has no audio | Video-only output (no error) |
| `audio.preserve: false` | Audio discarded |

Output duration is controlled by the **video timeline**, not the audio stream.
(`-shortest` is not used.)

---

## FP16 Behavior

FP16 is **disabled by default**.

The RIFE IFNet_m architecture uses PReLU activations, which are known to be
numerically unstable in FP16 on some GPU architectures (particularly older
Turing cards). Enabling FP16 with an incompatible GPU may produce NaN frames.

To enable FP16 (after verifying stability on your GPU):
```yaml
model:
  fp16: true
```

Or test first:
```bash
python -m frameforge --input test_clip.mp4 --output test_out.mp4
```
If output looks correct, add `fp16: true` to your config.

---

## CUDA Requirement

CUDA is **mandatory**. There is no CPU fallback.

If CUDA is not available, FrameForge will exit with a clear error message:
```
CUDA is not available. An NVIDIA GPU with the CUDA toolkit is required.
```

Verify your setup:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
nvidia-smi
```

---

## Troubleshooting

### `CUDA is not available`
- Install PyTorch with CUDA: https://pytorch.org/get-started/locally/
- Check drivers: `nvidia-smi`

### `checksum mismatch`
- Delete `~/.cache/frameforge/` and retry (will re-download)
- Verify network connection is not intercepting HTTPS

### `FFmpeg encoder crashed`
- Ensure FFmpeg is installed: `ffmpeg -version`
- Check disk space

### `CUDA out of memory`
- Use `--scale 0.5` to reduce flow estimation resolution
- Process a shorter clip first

---

## Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

Tests do NOT require a GPU or the model weights.

---

## License

The FrameForge AI pipeline code is MIT licensed.

The vendored RIFE model code (`frameforge/interpolation/models/rife/`) is from
[ECCV2022-RIFE](https://github.com/hzwer/ECCV2022-RIFE) / [Practical-RIFE](https://github.com/hzwer/Practical-RIFE)
by Zhewei Huang (hzwer). Licensed for **non-commercial use only**.
See `frameforge/interpolation/models/rife/LICENSE` if present, or the upstream repository.

---

## Attribution

- RIFE: [hzwer/Practical-RIFE](https://github.com/hzwer/Practical-RIFE)
  — Zhewei Huang, Tianyuan Zhang, Ling-Hao Han, Chuan Wang, Xilin Chen, Yujie Hu
- Paper: [Real-Time Intermediate Flow Estimation for Video Frame Interpolation (ECCV 2022)](https://arxiv.org/abs/2011.06294)
