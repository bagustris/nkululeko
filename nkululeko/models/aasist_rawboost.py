"""RawBoost waveform augmentation, ported from Tak, Kamble, Patino, Todisco,
Evans, "RawBoost: A Raw Data Boosting and Augmentation Method applied to
Automatic Speaker Verification Anti-Spoofing", ICASSP 2022 -- via the
reference implementation in TakHemlata/SSL_Anti-spoofing (RawBoost.py),
as vendored in the user's radar-SSL_AASIST fork.

Pure numpy/scipy, no new dependencies. Ported verbatim (only renamed to
snake_case module-private helpers and given an [AASIST]-config-driven
entry point) rather than reimplemented, since the exact noise-generation
behavior is what the paper's/upstream's published results depend on.
"""

import copy

import numpy as np
from scipy import signal


def _rand_range(x1, x2, integer):
    y = np.random.uniform(low=x1, high=x2, size=(1,))
    return int(y[0]) if integer else float(y[0])


def _norm_wav(x, always):
    if always:
        x = x / np.amax(abs(x))
    elif np.amax(abs(x)) > 1:
        x = x / np.amax(abs(x))
    return x


def _gen_notch_coeffs(
    n_bands, min_f, max_f, min_bw, max_bw, min_coeff, max_coeff, min_g, max_g, fs
):
    b = 1
    for _ in range(n_bands):
        fc = _rand_range(min_f, max_f, 0)
        bw = _rand_range(min_bw, max_bw, 0)
        c = _rand_range(min_coeff, max_coeff, 1)
        if c / 2 == int(c / 2):
            c = c + 1
        f1 = fc - bw / 2
        f2 = fc + bw / 2
        if f1 <= 0:
            f1 = 1 / 1000
        if f2 >= fs / 2:
            f2 = fs / 2 - 1 / 1000
        b = np.convolve(
            signal.firwin(c, [float(f1), float(f2)], window="hamming", fs=fs), b
        )
    g = _rand_range(min_g, max_g, 0)
    _, h = signal.freqz(b, 1, fs=fs)
    b = pow(10, g / 20) * b / np.amax(abs(h))
    return b


def _filter_fir(x, b):
    n = b.shape[0] + 1
    xpad = np.pad(x, (0, n), "constant")
    y = signal.lfilter(b, 1, xpad)
    y = y[int(n / 2) : int(y.shape[0] - n / 2)]
    return y


def lnl_convolutive_noise(
    x,
    n_f,
    n_bands,
    min_f,
    max_f,
    min_bw,
    max_bw,
    min_coeff,
    max_coeff,
    min_g,
    max_g,
    min_bias_lin_nonlin,
    max_bias_lin_nonlin,
    fs,
):
    """Linear and non-linear convolutive noise (RawBoost algo 1)."""
    y = [0] * x.shape[0]
    for i in range(n_f):
        g_lo, g_hi = min_g, max_g
        if i == 1:
            g_lo = min_g - min_bias_lin_nonlin
            g_hi = max_g - max_bias_lin_nonlin
        b = _gen_notch_coeffs(
            n_bands, min_f, max_f, min_bw, max_bw, min_coeff, max_coeff, g_lo, g_hi, fs
        )
        y = y + _filter_fir(np.power(x, i + 1), b)
    y = y - np.mean(y)
    return _norm_wav(y, 0)


def isd_additive_noise(x, p, g_sd):
    """Impulsive signal-dependent noise (RawBoost algo 2)."""
    beta = _rand_range(0, p, 0)
    y = copy.deepcopy(x)
    x_len = x.shape[0]
    n = int(x_len * (beta / 100))
    idx = np.random.permutation(x_len)[:n]
    f_r = np.multiply(
        (2 * np.random.rand(idx.shape[0])) - 1, (2 * np.random.rand(idx.shape[0])) - 1
    )
    r = g_sd * x[idx] * f_r
    y[idx] = x[idx] + r
    return _norm_wav(y, 0)


def ssi_additive_noise(
    x,
    snr_min,
    snr_max,
    n_bands,
    min_f,
    max_f,
    min_bw,
    max_bw,
    min_coeff,
    max_coeff,
    min_g,
    max_g,
    fs,
):
    """Stationary signal-independent noise (RawBoost algo 3)."""
    noise = np.random.normal(0, 1, x.shape[0])
    b = _gen_notch_coeffs(
        n_bands, min_f, max_f, min_bw, max_bw, min_coeff, max_coeff, min_g, max_g, fs
    )
    noise = _filter_fir(noise, b)
    noise = _norm_wav(noise, 1)
    snr = _rand_range(snr_min, snr_max, 0)
    noise = (
        noise / np.linalg.norm(noise, 2) * np.linalg.norm(x, 2) / 10.0 ** (0.05 * snr)
    )
    return x + noise


def apply_rawboost(feature, sr, cfg, algo):
    """Apply one of RawBoost's augmentation algorithms to a waveform.

    `algo` selects: 0 none, 1 LnL convolutive, 2 ISD additive, 3 SSI
    additive, 4 all three in series, 5 1+2, 6 1+3, 7 2+3, 8 1 and 2 in
    parallel -- matching upstream's process_Rawboost_feature() exactly.
    `cfg` is an AasistConfig (or anything exposing the same rawboost_*
    attributes read below).
    """
    lnl_args = (
        cfg.rawboost_n_f,
        cfg.rawboost_n_bands,
        cfg.rawboost_min_f,
        cfg.rawboost_max_f,
        cfg.rawboost_min_bw,
        cfg.rawboost_max_bw,
        cfg.rawboost_min_coeff,
        cfg.rawboost_max_coeff,
        cfg.rawboost_min_g,
        cfg.rawboost_max_g,
        cfg.rawboost_min_bias_lin_nonlin,
        cfg.rawboost_max_bias_lin_nonlin,
        sr,
    )
    isd_args = (cfg.rawboost_p, cfg.rawboost_g_sd)
    ssi_args = (
        cfg.rawboost_snr_min,
        cfg.rawboost_snr_max,
        cfg.rawboost_n_bands,
        cfg.rawboost_min_f,
        cfg.rawboost_max_f,
        cfg.rawboost_min_bw,
        cfg.rawboost_max_bw,
        cfg.rawboost_min_coeff,
        cfg.rawboost_max_coeff,
        cfg.rawboost_min_g,
        cfg.rawboost_max_g,
        sr,
    )

    if algo == 1:
        return lnl_convolutive_noise(feature, *lnl_args)
    if algo == 2:
        return isd_additive_noise(feature, *isd_args)
    if algo == 3:
        return ssi_additive_noise(feature, *ssi_args)
    if algo == 4:
        feature = lnl_convolutive_noise(feature, *lnl_args)
        feature = isd_additive_noise(feature, *isd_args)
        return ssi_additive_noise(feature, *ssi_args)
    if algo == 5:
        feature = lnl_convolutive_noise(feature, *lnl_args)
        return isd_additive_noise(feature, *isd_args)
    if algo == 6:
        feature = lnl_convolutive_noise(feature, *lnl_args)
        return ssi_additive_noise(feature, *ssi_args)
    if algo == 7:
        feature = isd_additive_noise(feature, *isd_args)
        return ssi_additive_noise(feature, *ssi_args)
    if algo == 8:
        feature1 = lnl_convolutive_noise(feature, *lnl_args)
        feature2 = isd_additive_noise(feature, *isd_args)
        return _norm_wav(feature1 + feature2, 0)
    return feature
