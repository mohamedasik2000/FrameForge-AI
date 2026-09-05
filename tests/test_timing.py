import pytest
import numpy as np
from frameforge.interpolation.timing import TimingScheduler

def test_timing_scheduler_basics():
    # 30fps output, source is roughly 10fps
    scheduler = TimingScheduler(target_fps=30.0)
    
    img0 = np.zeros((10, 10, 3), dtype=np.uint8)
    img1 = np.ones((10, 10, 3), dtype=np.uint8)
    img2 = np.ones((10, 10, 3), dtype=np.uint8) * 2
    
    # First frame (t=0s)
    outputs = list(scheduler.push(img0, 0.0))
    assert len(outputs) == 1
    assert outputs[0][2] == 0.0 # timestep
    
    # Second frame (t=0.1s).
    # target_fps = 30 -> step = 1/30
    # Current Target PTS interval: 1/30, 2/30, 3/30
    outputs = list(scheduler.push(img1, 0.1))
    
    assert len(outputs) == 3
    # 0.0, 0.1 -> delta=0.1.
    # 1/30 -> t = (1/30) / 0.1 = 1/3
    assert np.isclose(outputs[0][2], 1/3)
    # 2/30 -> t = (2/30) / 0.1 = 2/3
    assert np.isclose(outputs[1][2], 2/3)
    # 3/30 -> t = (3/30) / 0.1 = 1.0 (exact match)
    assert outputs[2][2] == 1.0
    
    # Third frame (t=0.2s)
    outputs = list(scheduler.push(img2, 0.2))
    assert len(outputs) == 3
