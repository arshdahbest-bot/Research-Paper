import math

import pytest

from eyemorse.pointing import PointingTask


def make(**kw):
    return PointingTask(n_targets=3, radius=40, screen_w=1600, screen_h=900, seed=1, **kw)


def test_targets_are_reproducible_and_on_screen():
    a, b = make(), make()
    assert a.target == b.target
    x, y = a.target
    assert 60 <= x <= 1540 and 60 <= y <= 840
    assert math.dist(a.target, (800, 450)) >= 200


def test_hit_miss_and_fitts():
    task = make()
    task.click(*task.target, t=0.5)  # before start: ignored
    task.start(1.0)
    tx, ty = task.target
    assert task.click(tx + 100, ty, 2.0) is None  # miss
    r = task.click(tx + 10, ty, 3.0)
    assert r["hit"] and r["misses"] == 1 and not r["first_click_hit"]
    assert r["movement_time_s"] == pytest.approx(2.0)
    assert r["first_click_s"] == pytest.approx(1.0)
    expected_id = math.log2(math.dist((800, 450), (tx, ty)) / 80 + 1)
    assert r["fitts_id_bits"] == pytest.approx(expected_id, abs=1e-3)
    assert r["throughput_bps"] == pytest.approx(expected_id / 2.0, abs=1e-3)
    assert task.index == 1


def test_timeout_and_summary():
    task = make(timeout=5.0)
    task.start(0.0)
    assert task.update(4.9) is None
    r = task.update(5.0)
    assert not r["hit"] and r["throughput_bps"] is None
    task.click(*task.target, t=6.0)
    task.click(*task.target, t=7.0)
    assert task.done
    s = task.summary()
    assert s["targets"] == 3
    assert s["hit_rate_pct"] == pytest.approx(200 / 3)
    assert s["first_click_hit_rate_pct"] == pytest.approx(200 / 3)
