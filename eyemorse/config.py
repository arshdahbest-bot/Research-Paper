"""All tunable parameters in one place.

Every value can be overridden from a JSON file (``--config``) or from the
command line, and the exact config used is saved with each session log so
experiments are reproducible.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass
class Config:
    # --- Camera / tracking -------------------------------------------------
    camera: int = 0
    frame_width: int = 640
    frame_height: int = 480
    model_path: str = "models/face_landmarker.task"

    # --- Blink -> Morse input ---------------------------------------------
    # "both"       : close both eyes (natural blinks are filtered by min_dot)
    # "wink-left"  : only a left-eye wink counts (natural blinks are ignored)
    # "wink-right" : only a right-eye wink counts
    input_mode: str = "both"
    close_threshold: float = 0.50  # closure score at which the eye counts as closed
    open_threshold: float = 0.30  # closure score at which it counts as open again
    min_dot: float = 0.12  # closures shorter than this are ignored (s)
    dash_threshold: float = 0.40  # closures >= this are a dash (s)
    long_threshold: float = 1.20  # closures >= this switch CURSOR <-> TYPE mode (s)
    very_long_threshold: float = 3.00  # closures >= this pause / resume everything (s)
    letter_gap: float = 0.90  # eyes-open time that ends a letter (s)
    word_gap: float = 0.0  # eyes-open time that inserts a space; 0 = off (s)
    adaptive_timing: bool = False  # learn the dot/dash boundary from the user

    # --- Gaze -> cursor ----------------------------------------------------
    smoothing_min_cutoff: float = 0.5  # One Euro filter: lower = smoother
    smoothing_beta: float = 0.005  # One Euro filter: higher = less lag on fast moves
    gaze_blink_gate: float = 0.35  # ignore gaze while either eye is this closed
    blink_rewind: float = 0.15  # on a blink, put the cursor back to where it was this long ago (s)
    gaze_hold_after_blink: float = 0.12  # ignore gaze briefly after the eyes reopen (s)
    dwell_click: float = 0.0  # click after resting the gaze this long; 0 = off (s)
    dwell_radius: int = 50  # px
    scroll_amount: int = 5
    calibration_grid: int = 3  # 3 -> 9-point calibration
    ridge_alpha: float = 0.01
    calibration_file: str = "calibration.json"

    # --- Misc --------------------------------------------------------------
    dry_run: bool = False  # never touch the real mouse / keyboard
    switch_key: str = ""  # e.g. "f9": hold a key/switch as an extra Morse channel
    log_dir: str = "logs"
    log_frames: bool = False  # per-frame CSV (large) for offline analysis
    screen_width_cm: float = 0.0  # set both of these to report error in degrees
    viewing_distance_cm: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        data = json.loads(Path(path).read_text())
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**data)
