"""Main application: fuse gaze (pointing) and blink Morse (selection / typing).

Modes
-----
CURSOR : the cursor follows your gaze; Morse codes are mouse commands.
TYPE   : the cursor stays put; Morse codes are typed as text.
PAUSED : nothing happens until you close your eyes for ``very_long_threshold``.

A closure of ``long_threshold`` switches between CURSOR and TYPE.
"""

from __future__ import annotations

import argparse
import math
import time
from collections import deque
from dataclasses import fields

from .blink import LONG, VERY_LONG, BlinkDetector, Closure, SymbolClassifier, closure_score
from .config import Config
from .controller import Controller, make_controller
from .filters import PointFilter
from .gaze import GazeMapper, gaze_features
from .morse import CURSOR, TYPE, Action, MorseDecoder, interpret, preview

MOUSE_ACTIONS = {
    "left_click": lambda c, cfg: c.click("left"),
    "double_click": lambda c, cfg: c.click("left", 2),
    "right_click": lambda c, cfg: c.click("right"),
    "scroll_down": lambda c, cfg: c.scroll(-cfg.scroll_amount),
    "scroll_up": lambda c, cfg: c.scroll(cfg.scroll_amount),
}


class EyeMorseApp:
    """All interaction logic. Camera-free, so it can be driven by tests."""

    def __init__(self, cfg: Config, controller: Controller, logger=None, trial=None):
        self.cfg = cfg
        self.controller = controller
        self.logger = logger
        self.trial = trial
        self.screen_w, self.screen_h = controller.screen_size()

        self.detector = BlinkDetector(cfg.close_threshold, cfg.open_threshold)
        self.classifier = SymbolClassifier(cfg.min_dot, cfg.dash_threshold, cfg.long_threshold,
                                           cfg.very_long_threshold, cfg.adaptive_timing)
        self.decoder = MorseDecoder(cfg.letter_gap, cfg.word_gap)
        self.filter = PointFilter(cfg.smoothing_min_cutoff, cfg.smoothing_beta)
        self.mapper: GazeMapper | None = None

        self.mode = TYPE if trial else CURSOR
        self.paused = False
        self.frozen = False
        self.dragging = False
        self.typed = ""
        self.last_action = ""
        self.closure = 0.0
        self.counts = {"symbols": 0, "chars": 0, "clicks": 0, "unknown": 0, "mode_switches": 0}

        self.cursor: tuple[float, float] | None = None
        self._history: deque[tuple[float, float, float]] = deque(maxlen=120)
        self._gated = True
        self._hold_until = 0.0
        self._dwell_anchor: tuple[float, float] | None = None
        self._dwell_t = 0.0
        self._dwell_fired = False

    # ------------------------------------------------------------------ setup
    def apply_calibration(self, data: dict) -> None:
        self.mapper = GazeMapper.from_dict(data["mapper"])
        blink = data.get("blink")
        if blink and data.get("input_mode") == self.cfg.input_mode:
            self.detector.close_threshold = blink["close_threshold"]
            self.detector.open_threshold = blink["open_threshold"]
        self._log(time.monotonic(), "calibration_loaded", data.get("validation", {}).get("mean_px", ""))

    # ------------------------------------------------------------------ input
    def on_frame(self, t: float, bl: float | None, br: float | None, landmarks=None) -> None:
        """One camera frame. bl/br are None when no face is visible."""
        if bl is None:
            self.closure = 0.0
            return
        self.closure = closure_score(bl, br, self.cfg.input_mode)
        closure = self.detector.update(self.closure, t)
        if closure is not None:
            self.on_closure(closure, "eye")
        if landmarks is not None:
            self._update_gaze(t, max(bl, br), landmarks)

    def on_closure(self, closure: Closure, source: str) -> None:
        symbol = self.classifier.classify(closure.duration)
        self._log(closure.end, "closure", f"{closure.duration:.3f}", f"{source}:{symbol}")
        if symbol == VERY_LONG:
            self.toggle_pause(closure.end)
        elif self.paused or symbol is None:
            return
        elif symbol == LONG:
            self.toggle_mode(closure.end)
        else:
            self.decoder.push(symbol, closure.end)
            self.counts["symbols"] += 1
            if self.trial and self.mode == TYPE:
                self.trial.on_symbol(closure.start)

    def tick(self, t: float, input_busy: bool) -> None:
        """Called every frame; finishes letters once the input has paused."""
        if input_busy or self.paused:
            return
        for code in self.decoder.update(t):
            action = interpret(code, self.mode)
            if action is not None:
                self.execute(action, t, code)

    # ---------------------------------------------------------------- actions
    def execute(self, action: Action, t: float, code: str = "") -> None:
        self._log(t, "action", f"{action.kind}:{action.value!r}", code)
        c = self.controller
        if action.kind == "type":
            c.type_text(action.value)
            self.typed += action.value
            self.counts["chars"] += 1
            if self.trial:
                self.trial.on_type(action.value)
            self.last_action = f"'{action.value}'  ({code.strip() or 'gap'})"
        elif action.kind == "key":
            c.press(action.value)
            self.last_action = action.value
            if action.value == "backspace":
                self.typed = self.typed[:-1]
                if self.trial:
                    self.trial.on_backspace()
            elif action.value == "enter":
                self.typed = ""
                if self.trial:
                    self._finish_trial(t)
        elif action.kind == "mouse":
            self.last_action = action.value
            if action.value == "toggle_freeze":
                self.frozen = not self.frozen
                self.filter.reset()
            elif action.value == "toggle_drag":
                self._set_drag(not self.dragging)
            else:
                MOUSE_ACTIONS[action.value](c, self.cfg)
                self.counts["clicks"] += 1
        else:
            self.counts["unknown"] += 1
            self.last_action = f"unknown {action.value}"

    def toggle_mode(self, t: float) -> None:
        self.decoder.clear()
        self._set_drag(False)
        self.mode = TYPE if self.mode == CURSOR else CURSOR
        self.filter.reset()
        self.counts["mode_switches"] += 1
        self.last_action = f"mode -> {self.mode}"
        self._log(t, "mode", self.mode)

    def toggle_pause(self, t: float) -> None:
        self.paused = not self.paused
        self.decoder.clear()
        self._set_drag(False)
        self.filter.reset()
        self.last_action = "paused" if self.paused else "resumed"
        self._log(t, "pause", int(self.paused))

    def _set_drag(self, on: bool) -> None:
        if on and not self.dragging:
            self.controller.mouse_down()
        elif not on and self.dragging:
            self.controller.mouse_up()
        self.dragging = on

    def _finish_trial(self, t: float) -> None:
        result = self.trial.on_enter(t)
        if result:
            self._log(t, "trial", result["wpm"], result)
            print(f"Trial {result['trial']}: {result['wpm']:.2f} WPM, "
                  f"{result['error_rate_pct']:.1f}% error, KSPC {result['kspc']:.2f}")

    # ------------------------------------------------------------------- gaze
    def _update_gaze(self, t: float, max_blink: float, landmarks) -> None:
        if self.mapper is None or self.mode != CURSOR or self.paused or self.frozen:
            return
        gated = max_blink >= self.cfg.gaze_blink_gate or self.detector.closed
        if gated:
            if not self._gated:
                self._rewind(t)
            self._gated = True
            return
        if self._gated:
            self._gated = False
            self._hold_until = t + self.cfg.gaze_hold_after_blink
        if t < self._hold_until:
            return
        if self.decoder.buffer:
            return  # a command is being blinked: keep the cursor on its target until it runs

        x, y = self.mapper.predict(gaze_features(landmarks))
        x, y = self.filter(float(x), float(y), t)
        x = min(max(x, 0.0), self.screen_w - 1.0)
        y = min(max(y, 0.0), self.screen_h - 1.0)
        self.cursor = (x, y)
        self._history.append((t, x, y))
        self.controller.move(int(x), int(y))
        self._dwell(t)

    def _rewind(self, t: float) -> None:
        """Eyelids drag the iris estimate down just before a blink: undo that."""
        target = t - self.cfg.blink_rewind
        past = [p for p in self._history if p[0] <= target]
        if past:
            _, x, y = past[-1]
            self.cursor = (x, y)
            self.controller.move(int(x), int(y))
            self.filter.reset()
            self.filter(x, y, t)

    def _dwell(self, t: float) -> None:
        if self.cfg.dwell_click <= 0 or self.cursor is None:
            return
        if self._dwell_anchor is None or math.dist(self.cursor, self._dwell_anchor) > self.cfg.dwell_radius:
            self._dwell_anchor, self._dwell_t, self._dwell_fired = self.cursor, t, False
        elif not self._dwell_fired and t - self._dwell_t >= self.cfg.dwell_click:
            self.controller.click("left")
            self.counts["clicks"] += 1
            self._dwell_fired = True
            self.last_action = "dwell click"
            self._log(t, "action", "mouse:'dwell_click'")

    # ------------------------------------------------------------------ misc
    def _log(self, t, kind, value="", detail="") -> None:
        if self.logger:
            self.logger.event(t, kind, value, detail)

    @property
    def display_mode(self) -> str:
        return "PAUSED" if self.paused else self.mode


# ====================================================================== CLI
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="eyemorse", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="JSON config file to start from")
    p.add_argument("--save-config", help="write the effective config to this file and exit")
    p.add_argument("--calibrate", action="store_true", help="force a new calibration")
    p.add_argument("--trial", metavar="PHRASES", help="run a text-entry experiment with this phrase file")
    p.add_argument("--trial-send", action="store_true", help="also type trial text into the OS")
    p.add_argument("--no-shuffle", action="store_true", help="keep trial phrases in file order")
    for f in fields(Config):
        flag = "--" + f.name.replace("_", "-")
        if isinstance(f.default, bool):
            p.add_argument(flag, action="store_true", default=None)
        else:
            p.add_argument(flag, type=type(f.default), default=None)
    return p


def config_from_args(args) -> Config:
    cfg = Config.load(args.config) if args.config else Config()
    for f in fields(Config):
        value = getattr(args, f.name)
        if value is not None:
            setattr(cfg, f.name, value)
    if cfg.input_mode not in ("both", "wink-left", "wink-right"):
        raise SystemExit("--input-mode must be both, wink-left or wink-right")
    return cfg


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    cfg = config_from_args(args)
    if args.save_config:
        cfg.save(args.save_config)
        print(f"Saved config to {args.save_config}")
        return

    import cv2

    from . import ui
    from .calibration import Aborted, load_calibration, run_calibration, save_calibration
    from .logger import SessionLogger
    from .tracker import FaceTracker
    from .trial import TrialSession

    trial = TrialSession.from_file(args.trial, shuffle=not args.no_shuffle) if args.trial else None
    controller = make_controller(cfg.dry_run or (trial is not None and not args.trial_send))
    screen_w, screen_h = controller.screen_size()

    tracker = FaceTracker(cfg.model_path, cfg.camera, cfg.frame_width, cfg.frame_height)
    logger = SessionLogger(cfg.log_dir, cfg.to_dict(), cfg.log_frames)
    app = EyeMorseApp(cfg, controller, logger, trial)
    switch = None
    if cfg.switch_key:
        from .switch import SwitchInput
        switch = SwitchInput(cfg.switch_key)

    def calibrate():
        try:
            data = run_calibration(tracker, cfg, screen_w, screen_h)
        except Aborted:
            print("Calibration cancelled.")
            return None
        save_calibration(data, cfg.calibration_file)
        logger.write_json("calibration.json", data)
        print(f"Calibration saved. Validation: {data['validation']}")
        return data

    calib = None if args.calibrate else load_calibration(cfg.calibration_file)
    if calib and calib.get("screen") != [screen_w, screen_h]:
        print("Screen size changed since last calibration; recalibrating.")
        calib = None
    if calib is None and trial is None:
        calib = calibrate()
    if calib:
        app.apply_calibration(calib)
        logger.write_json("calibration.json", calib)

    win = "EyeMorse"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    try:
        cv2.setWindowProperty(win, cv2.WND_PROP_TOPMOST, 1)
    except cv2.error:
        pass

    fps, last_t, show_help = 0.0, time.monotonic(), False
    session_start = last_t
    print("Running. Focus the preview window and press q or Esc to quit.")
    try:
        while True:
            t, frame, face = tracker.read()
            fps = 0.9 * fps + 0.1 / max(t - last_t, 1e-3)
            last_t = t

            if face is None:
                app.on_frame(t, None, None)
            else:
                app.on_frame(t, face.blink_left, face.blink_right, face.landmarks)
            busy = app.detector.closed
            if switch:
                for closure in switch.poll():
                    app.on_closure(closure, "switch")
                busy = busy or switch.pressed
            app.tick(t, busy)

            gx, gy = app.cursor or ("", "")
            logger.frame(t, face is not None, face.blink_left if face else 0, face.blink_right if face else 0,
                         app.closure, gx, gy, app.display_mode)

            closed_for = app.detector.closed_for(t) or (switch.held_for(t) if switch else 0.0)
            state = {
                "mode": app.display_mode, "frozen": app.frozen, "dragging": app.dragging,
                "dry_run": controller.__class__ is Controller, "calibrated": app.mapper is not None,
                "input_mode": cfg.input_mode + (f" + {cfg.switch_key}" if switch else ""), "fps": fps,
                "closure": app.closure, "closed": busy, "open_th": app.detector.open_threshold,
                "close_th": app.detector.close_threshold, "closed_for": closed_for,
                "pending_label": app.classifier.label(closed_for) if closed_for else "",
                "buffer": app.decoder.buffer, "buffer_preview": preview(app.decoder.buffer, app.mode),
                "last_action": app.last_action, "typed": app.typed, "show_help": show_help,
            }
            if trial:
                state.update(trial_target=trial.target if not trial.done else "(all trials done - press q)",
                             trial_typed=trial.typed, trial_no=f"{trial.index + 1}/{len(trial.phrases)}")
            cv2.imshow(win, ui.render(frame, face, state))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("h"):
                show_help = not show_help
            elif key == ord("m"):
                app.toggle_mode(t)
            elif key == ord("p"):
                app.toggle_pause(t)
            elif key == ord("f"):
                app.frozen = not app.frozen
            elif key == ord("c"):
                data = calibrate()
                if data:
                    app.apply_calibration(data)
    finally:
        app._set_drag(False)
        summary = {"duration_s": round(time.monotonic() - session_start, 2), **app.counts,
                   "validation": (calib or {}).get("validation", {})}
        if trial:
            summary["trial_summary"] = trial.summary()
            logger.write_json("trials.json", trial.results)
        logger.write_json("summary.json", summary)
        logger.close()
        if switch:
            switch.stop()
        tracker.close()
        cv2.destroyAllWindows()
        print(f"Session saved to {logger.dir}")
