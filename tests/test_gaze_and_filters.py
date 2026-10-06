import numpy as np
import pytest

from eyemorse.filters import OneEuroFilter
from eyemorse.gaze import EYES, GazeMapper, gaze_features, trim_outliers


def synthetic_face(iris_dx=0.0, iris_dy=0.0):
    lm = np.zeros((478, 3))
    for eye, cx in zip(EYES, (200.0, 300.0)):
        lm[eye["c0"], :2] = (cx - 20, 200)
        lm[eye["c1"], :2] = (cx + 20, 200)
        lm[eye["iris"], :2] = (cx + iris_dx, 200 + iris_dy)
    lm[1, :2] = (250, 260)
    return lm


def test_gaze_features_are_scale_invariant():
    a = gaze_features(synthetic_face(8, 3))
    b = gaze_features(synthetic_face(8, 3) * 2)
    assert np.allclose(a, b)
    assert a[0] == pytest.approx(8 / 40)
    assert a[1] == pytest.approx(3 / 40)


def test_mapper_recovers_linear_mapping():
    rng = np.random.default_rng(0)
    F = rng.normal(size=(300, 4))
    Y = np.column_stack([960 + 400 * F[:, 0], 540 + 300 * F[:, 1]])
    m = GazeMapper(alpha=1e-6).fit(F, Y)
    assert np.allclose(m.predict(F[0]), Y[0], atol=1.0)
    m2 = GazeMapper.from_dict(m.to_dict())
    assert np.allclose(m2.predict(F[:5]), m.predict(F[:5]))


def test_trim_outliers():
    F = np.vstack([np.random.default_rng(1).normal(size=(50, 2)), [[50, 50]]])
    mask = trim_outliers(F)
    assert not mask[-1] and mask[:-1].mean() > 0.9


def test_one_euro_smooths_jitter_but_follows_steps():
    f = OneEuroFilter(min_cutoff=0.5, beta=0.005)
    rng = np.random.default_rng(2)
    out = [f(100 + rng.normal(0, 10), i / 30) for i in range(60)]
    assert np.std(out[30:]) < 5
    for i in range(60, 120):
        y = f(800.0, i / 30)
    assert y == pytest.approx(800, abs=10)
