"""Operating-system output: move/click the mouse and type keys.

``Controller`` is the dry-run base class (it only records what it would do);
``PyAutoGuiController`` drives the real mouse and keyboard.
"""

from __future__ import annotations


def detect_screen_size() -> tuple[int, int]:
    try:
        import pyautogui

        w, h = pyautogui.size()
        return int(w), int(h)
    except Exception:
        return 1920, 1080


class Controller:
    def __init__(self):
        self.history: list[tuple] = []
        self._size: tuple[int, int] | None = None

    def screen_size(self) -> tuple[int, int]:
        if self._size is None:
            self._size = detect_screen_size()
        return self._size

    def position(self) -> tuple[int, int]:
        return 0, 0

    def move(self, x: int, y: int) -> None:
        pass  # too frequent to record

    def click(self, button: str = "left", clicks: int = 1) -> None:
        self.history.append(("click", button, clicks))

    def scroll(self, amount: int) -> None:
        self.history.append(("scroll", amount))

    def mouse_down(self) -> None:
        self.history.append(("mouse_down",))

    def mouse_up(self) -> None:
        self.history.append(("mouse_up",))

    def type_text(self, text: str) -> None:
        self.history.append(("type", text))

    def press(self, key: str) -> None:
        self.history.append(("press", key))


class PyAutoGuiController(Controller):
    def __init__(self):
        super().__init__()
        import pyautogui

        pyautogui.PAUSE = 0  # default 0.1 s sleep per call would stall the camera loop
        # The corner fail-safe would fire whenever you look at a screen corner;
        # use q / Esc in the preview window or a very long eye closure instead.
        pyautogui.FAILSAFE = False
        self.gui = pyautogui
        self._w, self._h = pyautogui.size()

    def screen_size(self) -> tuple[int, int]:
        return self._w, self._h

    def position(self) -> tuple[int, int]:
        p = self.gui.position()
        return p.x, p.y

    def move(self, x: int, y: int) -> None:
        x = min(max(int(x), 1), self._w - 2)
        y = min(max(int(y), 1), self._h - 2)
        self.gui.moveTo(x, y)

    def click(self, button: str = "left", clicks: int = 1) -> None:
        super().click(button, clicks)
        self.gui.click(button=button, clicks=clicks, interval=0.08)

    def scroll(self, amount: int) -> None:
        super().scroll(amount)
        self.gui.scroll(amount)

    def mouse_down(self) -> None:
        super().mouse_down()
        self.gui.mouseDown()

    def mouse_up(self) -> None:
        super().mouse_up()
        self.gui.mouseUp()

    def type_text(self, text: str) -> None:
        super().type_text(text)
        self.gui.write(text)

    def press(self, key: str) -> None:
        super().press(key)
        self.gui.press(key)


def make_controller(dry_run: bool) -> Controller:
    if dry_run:
        return Controller()
    try:
        return PyAutoGuiController()
    except Exception as exc:  # no display, missing permission, Wayland, ...
        print(f"[warn] Cannot control the OS ({exc}); running in dry-run mode.")
        return Controller()
