"""Unit tests for the ported RawBoost augmentation (nkululeko/models/aasist_rawboost.py).

Pure numpy/scipy, no network/GPU -- checks shape/dtype/finiteness for each
algorithm rather than exact values (RawBoost is randomized by design), plus
that algo=0 (default, "bare AASIST" with no augmentation) is a no-op.
"""

import dataclasses

import numpy as np
import pytest

from nkululeko.models import aasist_rawboost as rb


@dataclasses.dataclass
class _FakeCfg:
    rawboost_n_f: int = 5
    rawboost_n_bands: int = 5
    rawboost_min_f: int = 20
    rawboost_max_f: int = 8000
    rawboost_min_bw: int = 100
    rawboost_max_bw: int = 1000
    rawboost_min_coeff: int = 10
    rawboost_max_coeff: int = 100
    rawboost_min_g: int = 0
    rawboost_max_g: int = 0
    rawboost_min_bias_lin_nonlin: int = 5
    rawboost_max_bias_lin_nonlin: int = 20
    rawboost_p: int = 10
    rawboost_g_sd: int = 2
    rawboost_snr_min: int = 10
    rawboost_snr_max: int = 40


@pytest.fixture
def signal():
    rng = np.random.default_rng(0)
    return rng.uniform(-0.5, 0.5, size=16000).astype(np.float64)


class TestApplyRawboost:
    @pytest.mark.parametrize("algo", [1, 2, 3, 4, 5, 6, 7, 8])
    def test_returns_same_length_finite_signal(self, signal, algo):
        out = rb.apply_rawboost(signal, 16000, _FakeCfg(), algo)
        assert out.shape == signal.shape
        assert np.all(np.isfinite(out))

    def test_algo_zero_is_a_no_op(self, signal):
        out = rb.apply_rawboost(signal, 16000, _FakeCfg(), 0)
        assert np.array_equal(out, signal)

    def test_unknown_algo_falls_back_to_no_op(self, signal):
        out = rb.apply_rawboost(signal, 16000, _FakeCfg(), 99)
        assert np.array_equal(out, signal)


class TestNormWav:
    def test_always_normalizes_to_unit_peak(self):
        x = np.array([0.1, -0.4, 0.2])
        out = rb._norm_wav(x, always=True)
        assert np.isclose(np.amax(np.abs(out)), 1.0)

    def test_leaves_signal_within_range_untouched(self):
        x = np.array([0.1, -0.4, 0.2])
        out = rb._norm_wav(x, always=False)
        assert np.array_equal(out, x)

    def test_scales_down_signal_exceeding_range(self):
        x = np.array([0.5, -2.0, 1.0])
        out = rb._norm_wav(x, always=False)
        assert np.isclose(np.amax(np.abs(out)), 1.0)


class TestIsdAdditiveNoise:
    def test_preserves_shape_and_finiteness(self, signal):
        out = rb.isd_additive_noise(signal, p=10, g_sd=2)
        assert out.shape == signal.shape
        assert np.all(np.isfinite(out))
