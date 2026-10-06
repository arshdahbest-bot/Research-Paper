"""One Euro filter (Casiez, Roussel & Vogel, CHI 2012) for cursor smoothing.

It smooths heavily when the gaze is still (removing webcam jitter) and lightly
when it moves fast (keeping lag low).
"""

from __future__ import annotations

import math


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self) -> None:
        self._t: float | None = None
        self._x: float = 0.0
        self._dx: float = 0.0

    def __call__(self, x: float, t: float) -> float:
        if self._t is None:
            self._t, self._x, self._dx = t, x, 0.0
            return x
        dt = t - self._t
        if dt <= 0:
            return self._x
        self._t = t
        dx = (x - self._x) / dt
        self._dx += _alpha(self.d_cutoff, dt) * (dx - self._dx)
        cutoff = self.min_cutoff + self.beta * abs(self._dx)
        self._x += _alpha(cutoff, dt) * (x - self._x)
        return self._x


class PointFilter:
    """One Euro filter applied to x and y independently."""

    def __init__(self, min_cutoff: float, beta: float):
        self.fx = OneEuroFilter(min_cutoff, beta)
        self.fy = OneEuroFilter(min_cutoff, beta)

    def __call__(self, x: float, y: float, t: float) -> tuple[float, float]:
        return self.fx(x, t), self.fy(y, t)

    def reset(self) -> None:
        self.fx.reset()
        self.fy.reset()
