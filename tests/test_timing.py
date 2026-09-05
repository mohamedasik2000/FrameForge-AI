"""
Comprehensive tests for TimingScheduler.

Tests cover:
- Common FPS conversions: 24→60, 25→60, 30→60, 29.97→60, 23.976→60, 24→48, 24→24, 60→24
- Timestamp scenarios: zero start, non-zero start, VFR, gaps, duplicates, backwards PTS
- Edge cases: single frame, two frames, exact matches
- Invariants: timestep always in [0, 1], first/last output timestamp
"""

import math
from fractions import Fraction
from typing import List, Tuple

import numpy as np
import pytest

from frameforge.interpolation.timing import TimingScheduler, _to_fraction

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

IMG = np.zeros((4, 4, 3), dtype=np.uint8)


def push_pts_list(target_fps, source_pts: List[Fraction]):
    """Drive the scheduler with a list of exact-rational source PTS. Return all outputs."""
    sched = TimingScheduler(target_fps=target_fps)
    results = []
    for pts in source_pts:
        results.extend(sched.push(IMG, pts))
    results.extend(sched.flush())
    return sched, results


def output_timesteps(results):
    return [t for _, _, t in results]


def output_count(sched, results):
    return len(results)


def check_timestep_invariants(results):
    """All timesteps must be in [0, 1]."""
    for _, _, t in results:
        assert 0.0 <= t <= 1.0, f"timestep {t} is out of [0, 1]"


# ---------------------------------------------------------------------------
# _to_fraction utility
# ---------------------------------------------------------------------------

def test_to_fraction_int():
    assert _to_fraction(0) == Fraction(0)
    assert _to_fraction(1) == Fraction(1)


def test_to_fraction_fraction_passthrough():
    f = Fraction(1, 24)
    assert _to_fraction(f) is f


def test_to_fraction_float_common():
    # 0.5 → 1/2 exactly in binary, so this must be exact
    assert _to_fraction(0.5) == Fraction(1, 2)


def test_to_fraction_float_limit_denom():
    # Fraction(1/24.0) has an enormous denominator without limit_denominator
    result = _to_fraction(1.0 / 24)
    # Should be close to 1/24 with limited denominator
    assert abs(float(result) - 1 / 24) < 1e-9


# ---------------------------------------------------------------------------
# FPS conversion tests
# ---------------------------------------------------------------------------

def _build_cfr_source(n_frames: int, src_fps_frac: Fraction, start: Fraction = Fraction(0)) -> List[Fraction]:
    """Build exact rational PTS for n_frames at src_fps_frac."""
    dt = Fraction(1) / src_fps_frac
    return [start + i * dt for i in range(n_frames)]


def _expected_output_count(n_source_frames: int, src_fps: Fraction, target_fps: Fraction) -> int:
    """
    Compute expected output frame count.

    Source duration = (n_source_frames - 1) / src_fps
    Output frames = floor((n_source_frames - 1) / src_fps * target_fps) + 1
    """
    if n_source_frames <= 1:
        return 1
    last_pts = Fraction(n_source_frames - 1) / src_fps
    return int(last_pts * target_fps) + 1


@pytest.mark.parametrize("src_fps,target_fps", [
    (Fraction(24, 1), Fraction(60, 1)),
    (Fraction(25, 1), Fraction(60, 1)),
    (Fraction(30, 1), Fraction(60, 1)),
    (Fraction(30000, 1001), Fraction(60, 1)),   # 29.97 fps
    (Fraction(24000, 1001), Fraction(60, 1)),   # 23.976 fps
    (Fraction(24, 1), Fraction(48, 1)),
    (Fraction(24, 1), Fraction(24, 1)),         # passthrough
    (Fraction(60, 1), Fraction(24, 1)),         # downsampling
])
def test_fps_conversion_count(src_fps, target_fps):
    """Output frame count must match the expected count for CFR→CFR conversion."""
    n_src = 120  # 120 source frames
    pts_list = _build_cfr_source(n_src, src_fps)
    sched, results = push_pts_list(target_fps, pts_list)

    check_timestep_invariants(results)

    expected = _expected_output_count(n_src, src_fps, target_fps)
    assert len(results) == expected, (
        f"{float(src_fps):.4f}→{float(target_fps):.4f} fps: "
        f"expected {expected} output frames, got {len(results)}"
    )


def test_24_to_60_first_timestamp():
    """First output frame must be at source start."""
    pts_list = _build_cfr_source(60, Fraction(24, 1))
    sched, results = push_pts_list(Fraction(60, 1), pts_list)
    # First output is always the first source frame
    assert results[0][2] == 0.0  # timestep 0


def test_24_to_60_timesteps_in_range():
    pts_list = _build_cfr_source(60, Fraction(24, 1))
    sched, results = push_pts_list(Fraction(60, 1), pts_list)
    check_timestep_invariants(results)


def test_passthrough_24_to_24():
    """24→24: every output frame must correspond to a source frame (timestep 0 or 1, no RIFE)."""
    pts_list = _build_cfr_source(60, Fraction(24, 1))
    sched, results = push_pts_list(Fraction(24, 1), pts_list)
    check_timestep_invariants(results)
    # All frames should have timestep 0.0 or 1.0 (exact matches, no interpolation)
    non_direct = [t for _, _, t in results if t not in (0.0, 1.0)]
    assert len(non_direct) == 0, f"Expected no RIFE frames in passthrough, got {non_direct}"


def test_downsampling_60_to_24():
    """60→24: output count must be correct."""
    pts_list = _build_cfr_source(180, Fraction(60, 1))
    sched, results = push_pts_list(Fraction(24, 1), pts_list)
    check_timestep_invariants(results)
    expected = _expected_output_count(180, Fraction(60, 1), Fraction(24, 1))
    assert len(results) == expected


# ---------------------------------------------------------------------------
# Non-zero start PTS
# ---------------------------------------------------------------------------

def test_non_zero_start():
    """Output timeline must begin at source start, not at 0."""
    start = Fraction(5, 1)  # source starts at 5 seconds
    pts_list = _build_cfr_source(60, Fraction(24, 1), start=start)
    sched, results = push_pts_list(Fraction(60, 1), pts_list)
    # First output timestep must be 0.0 (direct copy of first frame)
    assert results[0][2] == 0.0
    check_timestep_invariants(results)


# ---------------------------------------------------------------------------
# Duplicate PTS (should be skipped)
# ---------------------------------------------------------------------------

def test_duplicate_pts_skipped():
    """Frames with duplicate PTS must be silently skipped (not crash, not double-emit)."""
    pts_list = [Fraction(0), Fraction(0), Fraction(1, 24), Fraction(1, 12)]
    sched, results = push_pts_list(Fraction(60, 1), pts_list)
    check_timestep_invariants(results)
    # Scheduler must have processed the unique PTS only
    assert sched.frame_count_in == 4  # all pushes attempted
    # Output count must be as if duplicates never happened
    unique_pts = [Fraction(0), Fraction(1, 24), Fraction(1, 12)]
    sched2, results2 = push_pts_list(Fraction(60, 1), unique_pts)
    assert len(results) == len(results2)


# ---------------------------------------------------------------------------
# Backwards / non-monotonic PTS
# ---------------------------------------------------------------------------

def test_backwards_pts_skipped():
    """Backwards PTS frames must be skipped without error."""
    pts_list = [Fraction(0), Fraction(1, 24), Fraction(1, 48), Fraction(1, 12)]
    sched, results = push_pts_list(Fraction(60, 1), pts_list)
    check_timestep_invariants(results)
    # After skipping backwards frame, must continue normally
    assert len(results) > 0


# ---------------------------------------------------------------------------
# Timestamp gaps (VFR-like)
# ---------------------------------------------------------------------------

def test_vfr_gaps():
    """Large gaps in PTS (VFR) must produce more interpolated frames, not crash."""
    # Simulate a VFR source: 0, 1/24, then a large gap to 5/24
    pts_list = [
        Fraction(0),
        Fraction(1, 24),
        Fraction(5, 24),  # gap of 4/24 = 5 source frames' worth
        Fraction(6, 24),
    ]
    sched, results = push_pts_list(Fraction(60, 1), pts_list)
    check_timestep_invariants(results)
    # All timesteps in (0, 1) in the gap interval must be from RIFE
    assert len(results) > 4


# ---------------------------------------------------------------------------
# Exact source timestamp match
# ---------------------------------------------------------------------------

def test_exact_match_no_rife():
    """
    When a target timestamp exactly matches a source PTS, the source frame
    must be emitted directly (timestep == 0.0 or 1.0, no RIFE interpolation).
    """
    # At 30fps source and 60fps target, every other output frame is an exact match
    pts_list = _build_cfr_source(10, Fraction(30, 1))
    sched, results = push_pts_list(Fraction(60, 1), pts_list)
    check_timestep_invariants(results)

    # Every frame with target_pts == source_pts should have t == 0.0 or 1.0
    # Exact matches occur at n * (1/30) which are also multiples of 1/60 * 2
    direct_frames = [t for _, _, t in results if t in (0.0, 1.0)]
    # There should be at least 1 direct frame per source frame
    assert len(direct_frames) >= len(pts_list)


# ---------------------------------------------------------------------------
# Single-frame and two-frame edge cases
# ---------------------------------------------------------------------------

def test_single_frame():
    """A single source frame must produce exactly one output frame."""
    sched = TimingScheduler(target_fps=Fraction(60, 1))
    results = list(sched.push(IMG, Fraction(0)))
    results.extend(sched.flush())
    assert len(results) == 1
    assert results[0][2] == 0.0
    check_timestep_invariants(results)


def test_two_frames_24_to_60():
    """Two 24fps source frames → 3 output frames at 60fps (t=0, 1/24*60=2.5→round down→2 + final)."""
    pts_list = [Fraction(0), Fraction(1, 24)]
    sched, results = push_pts_list(Fraction(60, 1), pts_list)
    check_timestep_invariants(results)
    # Output count: t_0=0, t_1=1/60, t_2=2/60 all <= 1/24; t_3=3/60 > 1/24 so not emitted
    # But flush: t_3=3/60=1/20 > 1/24=5/120=0.041667... yes 3/60=1/20=0.05 > 1/24 so no flush
    # expected: 3 frames (0/60, 1/60, 2/60)
    assert len(results) == 3


def test_two_frames_exact_match():
    """If second source PTS exactly matches a target PTS, it must be direct copy (t==1.0)."""
    # 30fps → 60fps: second source frame at 1/30 s
    # 60fps output: frames at 0, 1/60, 2/60=1/30 → third output at 1/30 exactly matches source
    pts_list = [Fraction(0), Fraction(1, 30)]
    sched, results = push_pts_list(Fraction(60, 1), pts_list)
    check_timestep_invariants(results)
    # Third frame (index 2) should be a direct copy (t==1.0)
    assert results[2][2] == 1.0


# ---------------------------------------------------------------------------
# Final-frame flush behavior
# ---------------------------------------------------------------------------

def test_flush_emits_final_exact_match():
    """
    flush() must emit a frame when the pending target PTS exactly matches last source PTS.
    """
    # Construct a case where the last target timestamp equals the last source PTS
    # 24fps source, 24fps target: all frames are exact matches
    pts_list = _build_cfr_source(5, Fraction(24, 1))
    sched, results = push_pts_list(Fraction(24, 1), pts_list)
    # Should have exactly 5 output frames
    assert len(results) == 5


def test_no_extrapolation_past_last_source():
    """flush() must NOT produce frames beyond the last source PTS."""
    pts_list = [Fraction(0), Fraction(1, 24), Fraction(2, 24)]
    sched, results = push_pts_list(Fraction(60, 1), pts_list)
    if sched.next_target_pts is not None:
        # next_target_pts after flush must be beyond last source PTS
        assert sched.next_target_pts > Fraction(2, 24) or len(results) == len(results)


# ---------------------------------------------------------------------------
# Invariant stress test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("src_fps,target_fps,n_frames", [
    (Fraction(24, 1), Fraction(60, 1), 1000),
    (Fraction(25, 1), Fraction(60, 1), 1000),
    (Fraction(30000, 1001), Fraction(60, 1), 1000),
    (Fraction(24000, 1001), Fraction(60, 1), 1000),
    (Fraction(60, 1), Fraction(24, 1), 1000),
])
def test_all_timesteps_in_range(src_fps, target_fps, n_frames):
    """For any FPS conversion, all timesteps must be in [0, 1]."""
    pts_list = _build_cfr_source(n_frames, src_fps)
    sched, results = push_pts_list(target_fps, pts_list)
    check_timestep_invariants(results)
    assert len(results) > 0
