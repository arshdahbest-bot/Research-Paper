"""Morse code tables, a timing-based decoder and the code -> action mapping.

Everything here is pure (time is passed in), so it is fully unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass

DOT = "."
DASH = "-"
WORD_GAP = " "  # token emitted by the decoder after a long enough pause

# ITU Morse (lower-case output).
MORSE_TO_CHAR = {
    ".-": "a", "-...": "b", "-.-.": "c", "-..": "d", ".": "e", "..-.": "f",
    "--.": "g", "....": "h", "..": "i", ".---": "j", "-.-": "k", ".-..": "l",
    "--": "m", "-.": "n", "---": "o", ".--.": "p", "--.-": "q", ".-.": "r",
    "...": "s", "-": "t", "..-": "u", "...-": "v", ".--": "w", "-..-": "x",
    "-.--": "y", "--..": "z",
    "-----": "0", ".----": "1", "..---": "2", "...--": "3", "....-": "4",
    ".....": "5", "-....": "6", "--...": "7", "---..": "8", "----.": "9",
    ".-.-.-": ".", "--..--": ",", "..--..": "?", ".----.": "'", "-.-.--": "!",
    "-..-.": "/", "-.--.": "(", "-.--.-": ")", ".-...": "&", "---...": ":",
    "-.-.-.": ";", "-...-": "=", ".-.-.": "+", "-....-": "-", "..--.-": "_",
    ".-..-.": '"', ".--.-.": "@",
}
CHAR_TO_MORSE = {v: k for k, v in MORSE_TO_CHAR.items()}

# Editing codes in TYPE mode (same layout as Gboard's Morse keyboard).
TYPE_COMMANDS = {
    "..--": "space",
    "----": "backspace",
    ".-.-": "enter",
}

# Mouse commands in CURSOR mode. They all start with a dash, and leading dots
# are stripped before lookup, so a stray natural blink never clicks anything.
CURSOR_COMMANDS = {
    "-": "left_click",
    "--": "double_click",
    "-.": "right_click",
    "---": "toggle_freeze",
    "-..": "scroll_down",
    "-.-": "scroll_up",
    "-.-.": "toggle_drag",
}

CURSOR = "CURSOR"
TYPE = "TYPE"


@dataclass(frozen=True)
class Action:
    kind: str  # "type" | "key" | "mouse" | "unknown"
    value: str = ""


def interpret(code: str, mode: str) -> Action | None:
    """Map one decoded Morse code (or WORD_GAP) to an action for ``mode``."""
    if mode == TYPE:
        if code == WORD_GAP or TYPE_COMMANDS.get(code) == "space":
            return Action("type", " ")
        if code in TYPE_COMMANDS:
            return Action("key", TYPE_COMMANDS[code])
        ch = MORSE_TO_CHAR.get(code)
        return Action("type", ch) if ch else Action("unknown", code)

    if code == WORD_GAP:
        return None
    stripped = code.lstrip(DOT)
    if not stripped:
        return None  # only dots -> treat as natural blinks
    cmd = CURSOR_COMMANDS.get(stripped)
    return Action("mouse", cmd) if cmd else Action("unknown", code)


def preview(buffer: str, mode: str) -> str:
    """What the current, unfinished symbol buffer would produce."""
    if not buffer:
        return ""
    action = interpret(buffer, mode)
    if action is None:
        return "(ignored)"
    if action.kind == "type":
        return "space" if action.value == " " else action.value
    if action.kind == "unknown":
        return "?"
    return action.value


class MorseDecoder:
    """Collects dots/dashes and emits a code once the input pauses."""

    def __init__(self, letter_gap: float, word_gap: float = 0.0):
        self.letter_gap = letter_gap
        self.word_gap = word_gap
        self.buffer = ""
        self._last = 0.0
        self._word_pending = False

    def push(self, symbol: str, t: float) -> None:
        """Add a symbol whose input ended at time ``t``."""
        if symbol not in (DOT, DASH):
            raise ValueError(symbol)
        self.buffer += symbol
        self._last = t
        self._word_pending = False

    def update(self, t: float) -> list[str]:
        """Call while the input is idle (eyes open). Returns finished codes."""
        out: list[str] = []
        if self.buffer and t - self._last >= self.letter_gap:
            out.append(self.buffer)
            self.buffer = ""
            self._word_pending = self.word_gap > 0
        if self._word_pending and t - self._last >= self.word_gap:
            out.append(WORD_GAP)
            self._word_pending = False
        return out

    def clear(self) -> None:
        self.buffer = ""
        self._word_pending = False
