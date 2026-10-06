import pytest

from eyemorse.metrics import kspc, levenshtein, msd_error_rate, wpm


def test_metrics():
    assert levenshtein("kitten", "sitting") == 3
    assert msd_error_rate("hello", "hello") == 0
    assert msd_error_rate("abcd", "abce") == pytest.approx(25)
    assert wpm("hello world", 60) == pytest.approx(2.0)
    assert kspc(20, "abcd") == 5
