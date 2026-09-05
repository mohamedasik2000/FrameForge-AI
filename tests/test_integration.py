import pytest
import os
import subprocess
import shutil
from frameforge.config import FrameForgeConfig
from frameforge.pipeline import Pipeline

@pytest.fixture
def test_workspace(tmp_path):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is not installed")
        
    os.makedirs(tmp_path / "input", exist_ok=True)
    os.makedirs(tmp_path / "output", exist_ok=True)
    
    input_video = str(tmp_path / "input" / "test_24fps.mp4")
    # Generate a simple 24fps 1-second video with a bouncing ball
    subprocess.check_call([
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", "testsrc=duration=1:size=640x360:rate=24",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        input_video
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    yield tmp_path, input_video

def test_pipeline_integration(test_workspace):
    workspace, input_video = test_workspace
    output_video = str(workspace / "output" / "output_60fps.mp4")
    
    config = FrameForgeConfig()
    config.processing.target_fps = 60.0
    config.processing.scale = 1.0
    # No audio in the synthetic video
    config.audio.preserve = False
    
    pipeline = Pipeline(
        input_path=input_video,
        output_path=output_video,
        config=config
    )
    
    # Process
    list(pipeline.process())
    
    assert os.path.exists(output_video)
    
    # Verify the output is exactly 60 FPS CFR
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate,nb_frames,duration",
        "-of", "csv=p=0", output_video
    ]
    output = subprocess.check_output(cmd, text=True).strip().split(",")
    # output: r_frame_rate, duration, nb_frames (order varies depending on ffprobe, usually r_frame_rate,duration,nb_frames but CSV outputs based on stream query order)
    # let's use json output for clearer parsing
    
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate,nb_frames",
        "-of", "json", output_video
    ]
    import json
    metadata = json.loads(subprocess.check_output(cmd, text=True))
    stream = metadata["streams"][0]
    
    r_frame_rate = stream["r_frame_rate"]
    assert r_frame_rate == "60/1"
    
    nb_frames = int(stream["nb_frames"])
    # 1 second of 60 FPS should have exactly 60 frames
    assert nb_frames == 60
