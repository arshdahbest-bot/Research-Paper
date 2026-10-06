from eyemorse.app import EyeMorseApp
from eyemorse.blink import Closure
from eyemorse.config import Config
from eyemorse.controller import Controller
from eyemorse.morse import CHAR_TO_MORSE, CURSOR, TYPE
from eyemorse.trial import TrialSession

DOT_S, DASH_S = 0.2, 0.6


def send(app, codes, t=0.0):
    """Blink out a sequence of Morse codes, returning the end time."""
    for code in codes:
        for sym in code:
            d = DOT_S if sym == "." else DASH_S
            app.on_closure(Closure(t, t + d), "test")
            t += d + 0.3
            app.tick(t, False)
        t += 1.0
        app.tick(t, False)
    return t


def test_typing_word_with_backspace_and_space():
    ctrl = Controller()
    app = EyeMorseApp(Config(), ctrl)
    app.on_closure(Closure(0, 1.5), "test")  # long closure -> TYPE
    assert app.mode == TYPE
    codes = [CHAR_TO_MORSE[c] for c in "hi"] + ["..--", CHAR_TO_MORSE["x"], "----"]
    send(app, codes, t=2.0)
    assert app.typed == "hi "
    assert ("press", "backspace") in ctrl.history


def test_cursor_commands_and_pause():
    ctrl = Controller()
    app = EyeMorseApp(Config(), ctrl)
    assert app.mode == CURSOR
    t = send(app, ["-", "--", "."])
    assert ctrl.history == [("click", "left", 1), ("click", "left", 2)]
    app.on_closure(Closure(t, t + 3.5), "test")  # pause
    send(app, ["-"], t + 4)
    assert len(ctrl.history) == 2 and app.paused


def test_trial_scoring():
    app = EyeMorseApp(Config(), Controller(), trial=TrialSession(["hi"], shuffle=False))
    assert app.mode == TYPE
    send(app, [CHAR_TO_MORSE["h"], CHAR_TO_MORSE["i"], ".-.-"])
    r = app.trial.results[0]
    assert r["transcribed"] == "hi" and r["error_rate_pct"] == 0
    assert r["symbols"] == 4 + 2 + 4 and r["wpm"] > 0


def test_cursor_stays_on_target_while_command_is_blinked():
    import numpy as np

    class StubMapper:
        target = (500.0, 400.0)

        def predict(self, features):
            return np.array(self.target)

    ctrl = Controller()
    app = EyeMorseApp(Config(smoothing_min_cutoff=1000.0), ctrl)  # ~no smoothing
    app.mapper = StubMapper()
    lm = np.zeros((478, 3))
    app.on_frame(0.0, 0.0, 0.0, lm)
    app.on_frame(0.5, 0.0, 0.0, lm)
    assert app.cursor == (500.0, 400.0)

    app.on_closure(Closure(1.0, 1.6), "test")  # dash -> command in progress
    StubMapper.target = (900.0, 100.0)  # gaze drifts while the eyes reopen
    app.on_frame(2.0, 0.0, 0.0, lm)
    assert app.cursor == (500.0, 400.0)  # cursor held

    app.tick(2.6, False)  # letter gap passed -> click fires at the held spot
    assert ctrl.history == [("click", "left", 1)]
    app.on_frame(3.0, 0.0, 0.0, lm)
    assert app.cursor[0] > 800  # and gaze tracking resumes
