"""Regression tests for simultaneous synthetic spectrogram scenes."""

import numpy as np
import pytest

from sigid.cv_scene_corpus import _simultaneous_pair_starts


@pytest.mark.parametrize("first,second,total", [
    (12000, 12000, 120000),
    (12000, 48000, 120000),
    (48000, 12000, 120000),
    (48000, 48000, 120000),
    (120000, 20000, 120000),
    (20000, 120000, 120000),
])
def test_simultaneous_pair_has_real_burst_overlap(first, second, total):
    rng = np.random.default_rng(100)
    for _ in range(100):
        start_a, start_b = _simultaneous_pair_starts(
            first, second, scene_sample_count=total, rng=rng,
        )
        assert 0 <= start_a <= total - first
        assert 0 <= start_b <= total - second
        shared = min(start_a + first, start_b + second) - max(start_a, start_b)
        assert shared >= (min(first, second) + 1) // 2


@pytest.mark.parametrize("first,second,total", [
    (0, 100, 1000), (100, 0, 1000), (1001, 100, 1000),
    (100, 1001, 1000),
])
def test_invalid_simultaneous_intervals_are_rejected(first, second, total):
    with pytest.raises(ValueError):
        _simultaneous_pair_starts(
            first, second, scene_sample_count=total, rng=np.random.default_rng(1),
        )
