"""
TimingScheduler — CFR output scheduling from arbitrary source timestamps.

Semantics
---------
Given a source video with frames at PTS p_0 < p_1 < ... < p_N,
the scheduler produces output frames at target timestamps:

    t_n = p_0 + n / target_fps    for n = 0, 1, 2, ...

until t_n > p_N.

For each target timestamp t_n:
- If t_n == p_k for some source frame k   → emit source frame directly (no RIFE)
- If p_{k-1} < t_n < p_k                 → RIFE interpolation between frames k-1 and k
                                            with timestep = (t_n - p_{k-1}) / (p_k - p_{k-1})

timestep is always in (0, 1) exclusive for interpolated frames,
and exactly 0.0 or 1.0 for source-matched frames.

Exact arithmetic
----------------
All timestamps are stored as fractions.Fraction computed from the stream's
integer PTS values and rational time_base.

  rational_pts = Fraction(pts_int) * Fraction(time_base_num, time_base_den)

When the caller provides pts as a Fraction, it is used as-is (exact).
When the caller provides pts as a float, it is converted via Fraction(pts).limit_denominator(1_000_000)
to avoid spurious large denominators from binary floating-point representation.

Pipeline contract
-----------------
- Call push(img, pts) for each decoded source frame in order.
- After the final source frame, call flush() to emit remaining output frames.
- Each yield is (img0, img1, timestep) where:
    - timestep == 0.0  → write img0 directly (first source frame or exact match)
    - timestep == 1.0  → write img1 directly (current source frame exact match)
    - 0 < timestep < 1 → run RIFE on img0, img1 with this timestep
"""

from fractions import Fraction
from typing import Any, Generator, Iterator, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)

# Maximum denominator when converting float PTS to Fraction.
# 1_000_000 preserves microsecond precision, which is more than sufficient
# for any real video time base (e.g., 1/90000, 1/12800, etc.).
_FLOAT_MAX_DENOM = 1_000_000


def _to_fraction(pts: Union[float, int, Fraction]) -> Fraction:
    """Convert a PTS value to Fraction without creating unbounded denominators."""
    if isinstance(pts, Fraction):
        return pts
    if isinstance(pts, int):
        return Fraction(pts)
    # float — limit denominator to avoid irrational binary fractions
    return Fraction(pts).limit_denominator(_FLOAT_MAX_DENOM)


class TimingScheduler:
    """
    Streaming, memory-efficient CFR output scheduler.

    All target timestamps are computed on-the-fly; no large list is pre-built.
    """

    def __init__(self, target_fps: Union[int, float, Fraction]):
        """
        Args:
            target_fps: Desired output frame rate. May be integer, float, or Fraction.
                        Examples: 60, 60.0, Fraction(60000, 1001) for 59.94 fps.
        """
        if isinstance(target_fps, Fraction):
            self.target_fps = target_fps
        else:
            self.target_fps = _to_fraction(target_fps)

        if self.target_fps <= 0:
            raise ValueError(f"target_fps must be positive, got {target_fps}")

        # Duration of one output frame
        self.target_frame_duration: Fraction = Fraction(1) / self.target_fps

        # State — reset on first push
        self._started: bool = False
        self._prev_img: Optional[Any] = None
        self._prev_pts: Optional[Fraction] = None
        self._next_target_pts: Optional[Fraction] = None  # next output PTS to emit

        # Counters for progress reporting
        self.frame_count_in: int = 0
        self.frame_count_out: int = 0

    def push(
        self,
        img: Any,
        pts: Union[float, int, Fraction],
    ) -> Generator[Tuple[Any, Any, float], None, None]:
        """
        Push a source frame.

        Yields (img0, img1, timestep) for each output frame whose target PTS
        falls within (prev_pts, curr_pts].

        timestep is a float in [0.0, 1.0].
        """
        curr_pts = _to_fraction(pts)
        self.frame_count_in += 1

        if not self._started:
            # First frame: start target timeline at this PTS
            self._started = True
            self._prev_img = img
            self._prev_pts = curr_pts
            self._next_target_pts = curr_pts  # first output timestamp = source start
            # Emit first output frame (exact match with source)
            yield (img, img, 0.0)
            self.frame_count_out += 1
            self._next_target_pts += self.target_frame_duration
            return

        # Reject non-monotonic / duplicate PTS
        if curr_pts <= self._prev_pts:
            logger.warning(
                f"Non-monotonic PTS: prev={float(self._prev_pts):.6f}s "
                f"current={float(curr_pts):.6f}s — frame skipped."
            )
            return

        interval = curr_pts - self._prev_pts  # always > 0

        # Emit all output timestamps t where prev_pts < t <= curr_pts
        while self._next_target_pts <= curr_pts:
            t = self._next_target_pts

            if t == curr_pts:
                # Exact match: emit current source frame directly
                timestep = 1.0
            elif t == self._prev_pts:
                # Exact match with previous: emit previous frame directly
                # (only possible on first iteration if target stayed behind)
                timestep = 0.0
            else:
                # Interpolate: compute exact rational timestep, clamp to [0, 1]
                ts_frac = (t - self._prev_pts) / interval
                # ts_frac is always in (0, 1) here because prev_pts < t < curr_pts
                timestep = float(ts_frac)
                # Safety clamp — should never trigger with exact rational math
                timestep = max(0.0, min(1.0, timestep))

            yield (self._prev_img, img, timestep)
            self.frame_count_out += 1
            self._next_target_pts += self.target_frame_duration

        self._prev_img = img
        self._prev_pts = curr_pts

    def flush(self) -> Generator[Tuple[Any, Any, float], None, None]:
        """
        Emit any remaining output frame that exactly matches the last source PTS.

        After all source frames have been pushed, the final source PTS itself
        may be the target of a pending output frame. This method handles that case.

        Note: flush() does NOT extrapolate beyond the last source PTS.
        Output is bounded by the source video's timeline.
        """
        if not self._started or self._prev_img is None:
            return

        # If the next target PTS exactly equals the last source PTS, emit it
        if self._next_target_pts == self._prev_pts:
            yield (self._prev_img, self._prev_img, 0.0)
            self.frame_count_out += 1
            self._next_target_pts += self.target_frame_duration

    @property
    def next_target_pts(self) -> Optional[Fraction]:
        """The next output PTS that will be emitted (for diagnostics)."""
        return self._next_target_pts
