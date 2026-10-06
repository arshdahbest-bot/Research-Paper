"""Per-session research logs: events.csv, optional frames.csv, summary.json."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path


class SessionLogger:
    def __init__(self, root: str | Path, config: dict, log_frames: bool = False):
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.dir = Path(root) / f"session_{stamp}"
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "config.json").write_text(json.dumps(config, indent=2))
        self.t0 = time.monotonic()

        self._events_file = open(self.dir / "events.csv", "w", newline="")
        self._events = csv.writer(self._events_file)
        self._events.writerow(["t", "event", "value", "detail"])

        self._frames_file = None
        if log_frames:
            self._frames_file = open(self.dir / "frames.csv", "w", newline="")
            self._frames = csv.writer(self._frames_file)
            self._frames.writerow(
                ["t", "face", "blink_left", "blink_right", "closure", "gaze_x", "gaze_y", "mode"]
            )

    def event(self, t: float, kind: str, value="", detail="") -> None:
        self._events.writerow([f"{t - self.t0:.4f}", kind, value, detail])

    def frame(self, t, face, bl, br, closure, gx, gy, mode) -> None:
        if self._frames_file:
            self._frames.writerow(
                [f"{t - self.t0:.4f}", int(face), f"{bl:.3f}", f"{br:.3f}",
                 f"{closure:.3f}", gx, gy, mode]
            )

    def write_json(self, name: str, data) -> None:
        (self.dir / name).write_text(json.dumps(data, indent=2))

    def close(self) -> None:
        self._events_file.close()
        if self._frames_file:
            self._frames_file.close()
