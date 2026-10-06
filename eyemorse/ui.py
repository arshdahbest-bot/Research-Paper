"""Preview window: mirrored camera, eye landmarks and a live status panel."""

from __future__ import annotations

import cv2
import numpy as np

from .gaze import EYES
from .morse import CURSOR_COMMANDS, MORSE_TO_CHAR, TYPE_COMMANDS

FONT = cv2.FONT_HERSHEY_SIMPLEX
PANEL_W = 380
CAM_W, CAM_H = 480, 360
MODE_COLORS = {"CURSOR": (80, 200, 120), "TYPE": (255, 170, 60), "PAUSED": (90, 90, 230)}


def _text(img, s, xy, scale=0.55, color=(235, 235, 235), thick=1):
    cv2.putText(img, s, xy, FONT, scale, color, thick, cv2.LINE_AA)


def render(frame, face, s) -> np.ndarray:
    """``s`` is a dict of display state assembled by the app."""
    h, w = frame.shape[:2]
    cam = frame.copy()
    if face is not None:
        for eye in EYES:
            for idx in (eye["c0"], eye["c1"]):
                cv2.circle(cam, tuple(face.landmarks[idx, :2].astype(int)), 2, (0, 255, 255), -1)
            cv2.circle(cam, tuple(face.landmarks[eye["iris"], :2].astype(int)), 3, (0, 0, 255), -1)
    cam = cv2.resize(cv2.flip(cam, 1), (CAM_W, CAM_H))
    if face is None:
        _text(cam, "NO FACE DETECTED", (110, 180), 0.8, (0, 0, 255), 2)
    if s.get("show_help"):
        _draw_help(cam, s["mode"])

    panel = np.full((CAM_H, PANEL_W, 3), 30, np.uint8)
    mode = s["mode"]
    _text(panel, mode, (12, 32), 0.9, MODE_COLORS.get(mode, (255, 255, 255)), 2)
    flags = [f for f, on in (("FROZEN", s["frozen"]), ("DRAG", s["dragging"]),
                             ("DRY-RUN", s["dry_run"]), ("NO CALIB", not s["calibrated"])) if on]
    _text(panel, " ".join(flags), (150, 30), 0.45, (120, 200, 255))
    _text(panel, f"input: {s['input_mode']}   {s['fps']:.0f} fps", (12, 56), 0.45, (170, 170, 170))

    # Closure bar with thresholds and the symbol the current closure would produce.
    x0, y0, bw, bh = 12, 70, PANEL_W - 24, 16
    cv2.rectangle(panel, (x0, y0), (x0 + bw, y0 + bh), (70, 70, 70), -1)
    fill = int(bw * min(max(s["closure"], 0.0), 1.0))
    cv2.rectangle(panel, (x0, y0), (x0 + fill, y0 + bh), (0, 200, 255) if s["closed"] else (150, 150, 150), -1)
    for th, col in ((s["open_th"], (0, 255, 0)), (s["close_th"], (0, 0, 255))):
        xt = x0 + int(bw * th)
        cv2.line(panel, (xt, y0 - 3), (xt, y0 + bh + 3), col, 2)
    if s["pending_label"]:
        _text(panel, f"{s['pending_label']}  {s['closed_for']:.2f}s", (x0, y0 + 38), 0.6, (0, 220, 255), 2)

    _text(panel, "Morse:", (12, 140), 0.55, (170, 170, 170))
    _text(panel, s["buffer"] or "-", (80, 142), 0.9, (255, 255, 255), 2)
    if s["buffer"]:
        _text(panel, f"-> {s['buffer_preview']}", (80, 170), 0.6, (0, 220, 255))

    _text(panel, "Last:", (12, 200), 0.5, (170, 170, 170))
    _text(panel, s["last_action"], (70, 200), 0.5)
    _text(panel, "Typed:", (12, 228), 0.5, (170, 170, 170))
    _text(panel, s["typed"][-28:] + "_", (12, 252), 0.6)

    if s.get("trial_target") is not None:
        _text(panel, f"TRIAL {s['trial_no']}:", (12, 286), 0.5, (255, 200, 120))
        _text(panel, s["trial_target"][:32], (12, 308), 0.55, (255, 255, 255))
        _text(panel, s["trial_typed"][-32:] + "_", (12, 330), 0.55, (0, 220, 255))
    else:
        _text(panel, "h help  c calibrate  m mode  p pause  q quit", (12, CAM_H - 12), 0.4, (150, 150, 150))
    return np.hstack([cam, panel])


def _draw_help(img, mode):
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (CAM_W, CAM_H), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
    lines = ["Close 1.2s: switch mode   Close 3s: pause"]
    if mode == "TYPE":
        lines += [f"{c:7s} {n}" for c, n in TYPE_COMMANDS.items()]
        items = sorted(((v, k) for k, v in MORSE_TO_CHAR.items() if v.isalpha()))
        lines += ["   ".join(f"{ch} {code}" for ch, code in items[i:i + 4]) for i in range(0, 26, 4)]
    else:
        lines += [f"{c:6s} {n}" for c, n in CURSOR_COMMANDS.items()]
    for i, line in enumerate(lines):
        _text(img, line, (10, 22 + i * 21), 0.45)
