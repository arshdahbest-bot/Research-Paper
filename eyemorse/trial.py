"""Text-entry trials: show a phrase, time the user typing it, score the result."""

from __future__ import annotations

import random
from pathlib import Path

from . import metrics


class TrialSession:
    def __init__(self, phrases: list[str], shuffle: bool = True, seed: int | None = None):
        self.phrases = list(phrases)
        if shuffle:
            random.Random(seed).shuffle(self.phrases)
        self.index = 0
        self.results: list[dict] = []
        self._reset()

    @classmethod
    def from_file(cls, path: str | Path, **kw) -> "TrialSession":
        lines = [l.strip().lower() for l in Path(path).read_text().splitlines()]
        return cls([l for l in lines if l and not l.startswith("#")], **kw)

    def _reset(self) -> None:
        self.typed = ""
        self.symbols = 0
        self.backspaces = 0
        self.start: float | None = None

    @property
    def done(self) -> bool:
        return self.index >= len(self.phrases)

    @property
    def target(self) -> str:
        return "" if self.done else self.phrases[self.index]

    def on_symbol(self, t: float) -> None:
        if self.start is None:
            self.start = t  # timing starts at the first dot/dash
        self.symbols += 1

    def on_type(self, text: str) -> None:
        self.typed += text

    def on_backspace(self) -> None:
        self.typed = self.typed[:-1]
        self.backspaces += 1

    def on_enter(self, t: float) -> dict | None:
        """Finish the current phrase; returns its scored result."""
        if self.done or self.start is None:
            return None
        seconds = t - self.start
        result = {
            "trial": self.index + 1,
            "presented": self.target,
            "transcribed": self.typed,
            "seconds": round(seconds, 3),
            "wpm": round(metrics.wpm(self.typed, seconds), 3),
            "error_rate_pct": round(metrics.msd_error_rate(self.target, self.typed), 2),
            "symbols": self.symbols,
            "kspc": round(metrics.kspc(self.symbols, self.typed), 3),
            "backspaces": self.backspaces,
        }
        self.results.append(result)
        self.index += 1
        self._reset()
        return result

    def summary(self) -> dict:
        if not self.results:
            return {}
        n = len(self.results)
        return {
            "trials": n,
            "mean_wpm": sum(r["wpm"] for r in self.results) / n,
            "mean_error_rate_pct": sum(r["error_rate_pct"] for r in self.results) / n,
            "mean_kspc": sum(r["kspc"] for r in self.results) / n,
        }
