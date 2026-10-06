"""Optional second input channel: hold a key / USB switch to send Morse.

Uses a global keyboard hook (pynput), so it works while another app has focus.
Press duration is classified exactly like an eye closure, which also makes it a
convenient baseline condition for experiments (switch vs. blink vs. wink).
"""

from __future__ import annotations

import queue
import time

from .blink import Closure


class SwitchInput:
    def __init__(self, key_name: str):
        from pynput import keyboard

        self._target = getattr(keyboard.Key, key_name.lower(), None) or keyboard.KeyCode.from_char(key_name)
        self._queue: queue.Queue[Closure] = queue.Queue()
        self._down_at: float | None = None
        self._listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self._listener.daemon = True
        self._listener.start()

    def _on_press(self, key):
        if key == self._target and self._down_at is None:
            self._down_at = time.monotonic()

    def _on_release(self, key):
        if key == self._target and self._down_at is not None:
            self._queue.put(Closure(self._down_at, time.monotonic()))
            self._down_at = None

    @property
    def pressed(self) -> bool:
        return self._down_at is not None

    def held_for(self, t: float) -> float:
        down = self._down_at
        return t - down if down is not None else 0.0

    def poll(self) -> list[Closure]:
        out = []
        while not self._queue.empty():
            out.append(self._queue.get_nowait())
        return out

    def stop(self) -> None:
        self._listener.stop()
