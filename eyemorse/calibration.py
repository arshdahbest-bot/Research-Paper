"""Full-screen calibration: gaze targets, blink thresholds and a validation pass.

The validation targets are *not* used for fitting, so the reported error is an
honest estimate of pointing accuracy (report it in your paper).
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

from .blink import closure_score
from .config import Config
from .gaze import GazeMapper, gaze_features, trim_outliers

WINDOW = "EyeMorse calibration"
SETTLE = 0.7  # s to let the eyes land on a target before sampling
SAMPLE = 1.0  # s of samples per target


class Aborted(Exception):
    pass


def grid_targets(n: int, margin: float = 0.08) -> list[tuple[float, float]]:
    vals = np.linspace(margin, 1 - margin, n)
    return [(x, y) for y in vals for x in vals]


VALIDATION_TARGETS = [(0.5, 0.5), (0.3, 0.3), (0.7, 0.3), (0.3, 0.7), (0.7, 0.7)]


class _Screen:
    def __init__(self, w: int, h: int):
        self.w, self.h = w, h
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def show(self, lines=(), target=None, progress=1.0, color=(255, 255, 255)) -> int:
        canvas = np.zeros((self.h, self.w, 3), np.uint8)
        if target is not None:
            c = (int(target[0] * self.w), int(target[1] * self.h))
            cv2.circle(canvas, c, int(10 + 30 * progress), color, 2, cv2.LINE_AA)
            cv2.circle(canvas, c, 6, (0, 0, 255), -1, cv2.LINE_AA)
        y = self.h // 2 - 30 * len(lines) // 2
        for line in lines:
            size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            cv2.putText(canvas, line, ((self.w - size[0]) // 2, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (230, 230, 230), 2, cv2.LINE_AA)
            y += 45
        cv2.imshow(WINDOW, canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            raise Aborted()
        return key

    def close(self):
        cv2.destroyWindow(WINDOW)


def _countdown(screen: _Screen, tracker, lines, seconds: float) -> None:
    end = time.monotonic() + seconds
    while (left := end - time.monotonic()) > 0:
        tracker.read()  # keep the camera / tracker warm
        if screen.show([*lines, f"{math.ceil(left)}", "(SPACE = start now, Esc = cancel)"]) == 32:
            return


def _collect_target(screen, tracker, target, gate):
    """Show a target, return the gaze feature vectors sampled while fixating."""
    t0 = time.monotonic()
    feats = []
    while (elapsed := time.monotonic() - t0) < SETTLE + SAMPLE:
        _, _, face = tracker.read()
        progress = max(0.0, 1 - elapsed / (SETTLE + SAMPLE))
        screen.show(target=target, progress=progress)
        if elapsed >= SETTLE and face is not None and max(face.blink_left, face.blink_right) < gate:
            feats.append(gaze_features(face.landmarks))
    return feats


def _collect_closure(tracker, screen, lines, seconds, input_mode):
    end = time.monotonic() + seconds
    vals = []
    while time.monotonic() < end:
        _, _, face = tracker.read()
        screen.show(lines)
        if face is not None:
            vals.append(closure_score(face.blink_left, face.blink_right, input_mode))
    return vals


def run_calibration(tracker, cfg: Config, screen_w: int, screen_h: int) -> dict:
    """Interactive calibration. Returns a dict (see save_calibration)."""
    screen = _Screen(screen_w, screen_h)
    try:
        _countdown(screen, tracker, [
            "Gaze calibration",
            "Keep your head still and look at the centre of each red dot.",
        ], 4)

        X, Y = [], []
        for target in grid_targets(cfg.calibration_grid):
            feats = _collect_target(screen, tracker, target, cfg.gaze_blink_gate)
            if not feats:
                continue
            F = np.array(feats)
            F = F[trim_outliers(F)]
            X.extend(F)
            Y.extend([(target[0] * screen_w, target[1] * screen_h)] * len(F))
        if len(X) < 20:
            raise RuntimeError("Too few gaze samples - is your face visible and well lit?")
        mapper = GazeMapper(alpha=cfg.ridge_alpha).fit(np.array(X), np.array(Y))

        # Validation on targets not used for fitting.
        errors = []
        for target in VALIDATION_TARGETS:
            feats = _collect_target(screen, tracker, target, cfg.gaze_blink_gate)
            if feats:
                pred = np.median(mapper.predict(np.array(feats)), axis=0)
                truth = np.array([target[0] * screen_w, target[1] * screen_h])
                errors.append(float(np.linalg.norm(pred - truth)))
        validation = accuracy_report(errors, screen_w, cfg)

        # Blink thresholds, personalised to this user / lighting / input mode.
        what = {"both": "CLOSE BOTH EYES", "wink-left": "WINK YOUR LEFT EYE",
                "wink-right": "WINK YOUR RIGHT EYE"}[cfg.input_mode]
        opened = _collect_closure(tracker, screen, ["Blink calibration",
                                  "Look at the screen with eyes relaxed and open..."], 2.0, cfg.input_mode)
        _countdown(screen, tracker, [f"Next: {what} for about 3 seconds, then open."], 3)
        closed = _collect_closure(tracker, screen, [f"{what} now (about 3 s)"], 4.0, cfg.input_mode)
        thresholds = blink_thresholds(opened, closed)

        screen.show(["Calibration done.",
                     f"Mean validation error: {validation.get('mean_px', float('nan')):.0f} px"])
        cv2.waitKey(1500)
    finally:
        screen.close()

    return {
        "screen": [screen_w, screen_h],
        "input_mode": cfg.input_mode,
        "mapper": mapper.to_dict(),
        "validation": validation,
        "blink": thresholds,
        "n_samples": len(X),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def blink_thresholds(opened: list[float], closed: list[float]) -> dict | None:
    """Place hysteresis thresholds between the user's open and closed levels."""
    if not opened or not closed:
        return None
    lo = float(np.median(opened))
    hi = float(np.percentile(closed, 90))
    if hi - lo < 0.2:
        return None  # not enough contrast; keep the defaults
    return {
        "open_level": lo,
        "closed_level": hi,
        "close_threshold": lo + 0.55 * (hi - lo),
        "open_threshold": lo + 0.30 * (hi - lo),
    }


def accuracy_report(errors_px: list[float], screen_w: int, cfg: Config) -> dict:
    if not errors_px:
        return {}
    e = np.array(errors_px)
    report = {"n_targets": len(e), "mean_px": float(e.mean()), "sd_px": float(e.std()),
              "max_px": float(e.max())}
    if cfg.screen_width_cm > 0 and cfg.viewing_distance_cm > 0:
        cm = e * cfg.screen_width_cm / screen_w
        deg = np.degrees(np.arctan2(cm, cfg.viewing_distance_cm))
        report.update(mean_deg=float(deg.mean()), sd_deg=float(deg.std()))
    return report


def save_calibration(data: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(data, indent=2))


def load_calibration(path: str | Path) -> dict | None:
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None
