"""Tests for the audiokit-backed energy VAD segmenter.

Unlike the Silero/pyannote backends (which need model downloads and are mocked
elsewhere), the energy segmenter is pure NumPy + audiokit, so it can be
exercised for real against a small synthetic WAV.
"""

import numpy as np
import pytest

audiofile = pytest.importorskip("audiofile")
pytest.importorskip("audiokit")

from nkululeko.segmenting.seg_energy import Energy_segmenter


def _write_burst_wav(path, sr=16000):
    """Silence, a loud 200 Hz burst, then silence — one detectable event."""
    t = np.arange(int(0.3 * sr)) / sr
    burst = np.sin(2 * np.pi * 200 * t).astype(np.float32)
    pre = np.zeros(int(0.5 * sr), dtype=np.float32)
    post = np.zeros(int(0.5 * sr), dtype=np.float32)
    sig = np.concatenate([pre, burst, post])
    audiofile.write(str(path), sig, sr)
    return sig, sr


def test_detects_single_event(tmp_path):
    wav = tmp_path / "burst.wav"
    _write_burst_wav(wav)

    seg = Energy_segmenter(not_testing=False)
    index = seg.get_segmentation_simple((str(wav),))

    # audformat segmented index: at least one (file, start, end) entry
    assert len(index) >= 1
    files = index.get_level_values("file")
    assert all(f == str(wav) for f in files)
    starts = index.get_level_values("start")
    ends = index.get_level_values("end")
    # the detected event overlaps the burst (0.5s–0.8s)
    assert any(s.total_seconds() <= 0.7 <= e.total_seconds() for s, e in zip(starts, ends))


def test_silence_falls_back_to_whole_file(tmp_path):
    wav = tmp_path / "silence.wav"
    sr = 16000
    audiofile.write(str(wav), np.zeros(sr, dtype=np.float32), sr)

    seg = Energy_segmenter(not_testing=False)
    index = seg.get_segmentation_simple((str(wav),))

    # No events detected -> one whole-file segment, never zero rows.
    assert len(index) == 1
    start = index.get_level_values("start")[0].total_seconds()
    end = index.get_level_values("end")[0].total_seconds()
    assert start == pytest.approx(0.0)
    assert end == pytest.approx(1.0, abs=0.05)
