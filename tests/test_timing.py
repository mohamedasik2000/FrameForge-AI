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
    outputs = scheduler.push(img0, 0.0)
    assert len(outputs) == 1
    assert outputs[0][2] == 0.0 # timestep
    
    # Second frame (t=0.1s). Output should hit: 0.033, 0.066, 0.1
    # 0.0, 0.1 -> delta=0.1.
    # We output pts: 0.0333... (t=0.33), 0.0666... (t=0.66)
    outputs = scheduler.push(img1, 0.1)
    
    assert len(outputs) == 3
    assert np.isclose(outputs[0][2], 1/3)
    assert np.isclose(outputs[1][2], 2/3)
    assert outputs[2][2] == 0.0 # Next source frame exactly hits a grid point or close to it
    
    # Third frame (t=0.2s)
    outputs = scheduler.push(img2, 0.2)
    assert len(outputs) == 3
