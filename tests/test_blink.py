import pytest

from eyemorse.blink import LONG, VERY_LONG, BlinkDetector, SymbolClassifier, closure_score
from eyemorse.morse import DASH, DOT


def feed(det, samples, dt=1 / 30):
    out = []
    for i, s in enumerate(samples):
        c = det.update(s, i * dt)
        if c:
            out.append(c)
    return out


def test_hysteresis():
    det = BlinkDetector(0.5, 0.3)
    # 0.4 never reaches close threshold; jitter between 0.35-0.6 stays closed
    closures = feed(det, [0.1, 0.4, 0.1, 0.6, 0.35, 0.6, 0.2, 0.1])
    assert len(closures) == 1
    assert closures[0].duration == pytest.approx(3 / 30)


def test_classifier():
    c = SymbolClassifier(min_dot=0.12, dash_threshold=0.4, long_threshold=1.2, very_long_threshold=3.0)
    assert c.classify(0.05) is None
    assert c.classify(0.2) == DOT
    assert c.classify(0.5) == DASH
    assert c.classify(1.5) == LONG
    assert c.classify(3.5) == VERY_LONG


def test_adaptive_threshold_moves_to_midpoint():
    c = SymbolClassifier(0.12, 0.4, 1.2, 3.0, adaptive=True)
    for _ in range(5):
        c.classify(0.30)  # this user's dots
        c.classify(0.70)  # and dashes
    assert c.dash_threshold == pytest.approx(0.5)


def test_wink_cancels_bilateral_blink():
    assert closure_score(0.9, 0.9, "wink-left") == 0
    assert closure_score(0.9, 0.1, "wink-left") == pytest.approx(0.8)
    assert closure_score(0.9, 0.1, "both") == pytest.approx(0.1)
