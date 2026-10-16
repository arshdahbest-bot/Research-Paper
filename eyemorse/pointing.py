"""Pointing (target selection) task for measuring gaze + blink clicking.

A full-screen window shows one circular target at a time at a random position.
The participant moves the cursor with their gaze and clicks with the Morse
click code. The click is a real OS click, received by the window, so the task
measures the whole system exactly as it would be used.

Per target it records movement time, misses and Fitts' law index of
difficulty (ID = log2(D / W + 1), Shannon formulation) and throughput
(ID / MT), as in ISO 9241-9 style pointing studies (nominal width).
"""

from __future__ import annotations

import math
import random

import numpy as np


class PointingTask:
    """Task logic only (no window), so it can be unit-tested."""

    def __init__(self, n_targets: int, radius: int, screen_w: int, screen_h: int,
                 timeout: float = 15.0, seed: int | None = None, min_move: int = 200):
        self.n_targets = n_targets
        self.radius = radius
        self.w, self.h = screen_w, screen_h
        self.timeout = timeout
        self.rng = random.Random(seed)
        self.min_move = min_move
        self.results: list[dict] = []
        self.started = False
        self.index = 0
        self.prev = (screen_w / 2, screen_h / 2)  # movement starts from the centre
        self.target = self._next_position()
        self.shown_at = 0.0
        self.misses = 0
        self.first_click: float | None = None

    def _next_position(self) -> tuple[float, float]:
        m = self.radius + 20
        for _ in range(100):
            p = (self.rng.uniform(m, self.w - m), self.rng.uniform(m, self.h - m))
            if math.dist(p, self.prev) >= self.min_move:
                return p
        return p

    @property
    def done(self) -> bool:
        return self.index >= self.n_targets

    def start(self, t: float) -> None:
        self.started = True
        self.shown_at = t

    def click(self, x: float, y: float, t: float) -> dict | None:
        """A click at (x, y). Returns the target's result if this click hit it."""
        if not self.started or self.done:
            return None
        if self.first_click is None:
            self.first_click = t
        if math.dist((x, y), self.target) <= self.radius:
            return self._finish(t, hit=True, click=(x, y))
        self.misses += 1
        return None

    def update(self, t: float) -> dict | None:
        """Call every frame; ends the target when it times out."""
        if self.started and not self.done and t - self.shown_at >= self.timeout:
            return self._finish(t, hit=False, click=None)
        return None

    def _finish(self, t: float, hit: bool, click) -> dict:
        distance = math.dist(self.prev, self.target)
        width = 2 * self.radius
        mt = t - self.shown_at
        fitts_id = math.log2(distance / width + 1)
        result = {
            "target": self.index + 1,
            "x": round(self.target[0]), "y": round(self.target[1]),
            "radius_px": self.radius,
            "distance_px": round(distance, 1),
            "hit": hit,
            "movement_time_s": round(mt, 3),
            "first_click_s": round(self.first_click - self.shown_at, 3) if self.first_click else None,
            "misses": self.misses,
            "first_click_hit": hit and self.misses == 0,
            "click_error_px": round(math.dist(click, self.target), 1) if click else None,
            "fitts_id_bits": round(fitts_id, 3),
            "throughput_bps": round(fitts_id / mt, 3) if hit and mt > 0 else None,
        }
        self.results.append(result)
        self.index += 1
        self.prev = self.target
        self.target = self._next_position()
        self.shown_at = t
        self.misses = 0
        self.first_click = None
        return result

    def summary(self) -> dict:
        if not self.results:
            return {}
        hits = [r for r in self.results if r["hit"]]
        tp = [r["throughput_bps"] for r in hits if r["throughput_bps"]]
        return {
            "targets": len(self.results),
            "hit_rate_pct": 100 * len(hits) / len(self.results),
            "first_click_hit_rate_pct": 100 * sum(r["first_click_hit"] for r in self.results) / len(self.results),
            "mean_movement_time_s": float(np.mean([r["movement_time_s"] for r in hits])) if hits else None,
            "mean_misses": float(np.mean([r["misses"] for r in self.results])),
            "mean_throughput_bps": float(np.mean(tp)) if tp else None,
        }


class PointingWindow:
    """Full-screen OpenCV window that draws the task and receives real clicks."""

    NAME = "EyeMorse pointing task"
    COUNTDOWN = 3.0

    def __init__(self, task: PointingTask):
        import cv2

        self.cv2 = cv2
        self.task = task
        self._clicks: list[tuple[int, int]] = []
        self._opened_at: float | None = None
        cv2.namedWindow(self.NAME, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self.NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.setMouseCallback(self.NAME, self._on_mouse)

    def _on_mouse(self, event, x, y, flags, param):
        if event == self.cv2.EVENT_LBUTTONDOWN:
            self._clicks.append((x, y))

    def step(self, t: float, status: str) -> list[dict]:
        """Process clicks / timeouts, draw a frame. Returns finished targets."""
        cv2, task = self.cv2, self.task
        if self._opened_at is None:
            self._opened_at = t
        if not task.started and t - self._opened_at >= self.COUNTDOWN:
            task.start(t)
            self._clicks.clear()  # ignore clicks made during the countdown

        finished = []
        for x, y in self._clicks:
            r = task.click(x, y, t)
            if r:
                finished.append(r)
        self._clicks.clear()
        r = task.update(t)
        if r:
            finished.append(r)

        canvas = np.full((task.h, task.w, 3), 24, np.uint8)
        white = (235, 235, 235)
        if not task.started:
            left = math.ceil(self.COUNTDOWN - (t - self._opened_at))
            self._centered(canvas, "Look at each red target and blink a dash ( - ) to click it.", -30, white)
            self._centered(canvas, f"Starting in {left}", 30, white)
        elif task.done:
            s = task.summary()
            self._centered(canvas, "Pointing task complete. Press q to finish.", -30, white)
            self._centered(canvas, f"Hit rate {s['hit_rate_pct']:.0f}%", 30, white)
        else:
            c = (int(task.target[0]), int(task.target[1]))
            cv2.circle(canvas, c, task.radius, (40, 40, 220), -1, cv2.LINE_AA)
            cv2.circle(canvas, c, 4, white, -1, cv2.LINE_AA)
            cv2.putText(canvas, f"Target {task.index + 1}/{task.n_targets}", (20, task.h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 1, cv2.LINE_AA)
        cv2.putText(canvas, status, (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2, cv2.LINE_AA)
        cv2.imshow(self.NAME, canvas)
        return finished

    def _centered(self, canvas, text, dy, color):
        cv2 = self.cv2
        size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        org = ((canvas.shape[1] - size[0]) // 2, canvas.shape[0] // 2 + dy)
        cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

    def close(self) -> None:
        self.cv2.destroyWindow(self.NAME)
