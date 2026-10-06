"""Turn a per-frame eye-closure signal into timed closures and Morse symbols."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from statistics import mean

from .morse import DASH, DOT

LONG = "LONG"  # switch mode
VERY_LONG = "VERY_LONG"  # pause / resume


@dataclass(frozen=True)
class Closure:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def closure_score(blink_left: float, blink_right: float, input_mode: str) -> float:
    """Combine per-eye closure (0 = open, 1 = shut) into one input signal."""
    if input_mode == "both":
        return min(blink_left, blink_right)  # both eyes must close
    if input_mode == "wink-left":
        return max(0.0, blink_left - blink_right)  # bilateral blinks cancel out
    if input_mode == "wink-right":
        return max(0.0, blink_right - blink_left)
    raise ValueError(f"unknown input_mode {input_mode!r}")


class BlinkDetector:
    """Hysteresis thresholding: closed above close_th, open again below open_th."""

    def __init__(self, close_threshold: float, open_threshold: float):
        if open_threshold >= close_threshold:
            raise ValueError("open_threshold must be below close_threshold")
        self.close_threshold = close_threshold
        self.open_threshold = open_threshold
        self.closed = False
        self.closed_since: float | None = None

    def update(self, score: float, t: float) -> Closure | None:
        """Feed one sample. Returns a Closure when the eye reopens."""
        if not self.closed and score >= self.close_threshold:
            self.closed = True
            self.closed_since = t
        elif self.closed and score <= self.open_threshold:
            closure = Closure(self.closed_since, t)
            self.closed = False
            self.closed_since = None
            return closure
        return None

    def closed_for(self, t: float) -> float:
        return t - self.closed_since if self.closed else 0.0

    def reset(self) -> None:
        self.closed = False
        self.closed_since = None


class SymbolClassifier:
    """Duration -> DOT / DASH / LONG / VERY_LONG (or None for noise).

    With ``adaptive=True`` the dot/dash boundary drifts to the midpoint between
    the user's recent mean dot and mean dash durations (1-D two-means), which
    lets the system follow a user who speeds up or tires during a session.
    """

    def __init__(
        self,
        min_dot: float,
        dash_threshold: float,
        long_threshold: float,
        very_long_threshold: float,
        adaptive: bool = False,
        window: int = 20,
    ):
        self.min_dot = min_dot
        self.dash_threshold = dash_threshold
        self.long_threshold = long_threshold
        self.very_long_threshold = very_long_threshold
        self.adaptive = adaptive
        self._dots: deque[float] = deque(maxlen=window)
        self._dashes: deque[float] = deque(maxlen=window)

    def classify(self, duration: float) -> str | None:
        if duration < self.min_dot:
            return None
        if duration >= self.very_long_threshold:
            return VERY_LONG
        if duration >= self.long_threshold:
            return LONG
        symbol = DASH if duration >= self.dash_threshold else DOT
        if self.adaptive:
            self._learn(symbol, duration)
        return symbol

    def _learn(self, symbol: str, duration: float) -> None:
        (self._dashes if symbol == DASH else self._dots).append(duration)
        if len(self._dots) >= 3 and len(self._dashes) >= 3:
            mid = (mean(self._dots) + mean(self._dashes)) / 2
            lo, hi = self.min_dot * 1.5, self.long_threshold * 0.8
            self.dash_threshold = min(max(mid, lo), hi)

    def label(self, duration: float) -> str:
        """Human-readable name of what a closure of this length would be."""
        if duration < self.min_dot:
            return ""
        if duration >= self.very_long_threshold:
            return "PAUSE"
        if duration >= self.long_threshold:
            return "MODE"
        return "DASH" if duration >= self.dash_threshold else "DOT"
