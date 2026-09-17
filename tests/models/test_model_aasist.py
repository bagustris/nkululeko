"""Unit tests for AasistModel (nkululeko/models/model_aasist.py).

Two levels, matching test_model_adm.py's approach for a model whose real
__init__ needs a full experiment context and downloads a real SSL
checkpoint:

1. _WaveformDataset against real (tiny, synthetic) WAV files on disk --
   the genuinely new integration surface (raw audio I/O, NaT handling,
   RawBoost hook, pad/truncate), independent of the network architecture.
2. AasistModel.evaluate()/get_probas() with __init__ patched out and
   attributes injected directly (mirroring test_model_adm.py's adm_model
   fixture), using a trivial stand-in for self.net so these tests don't
   need the real AASIST backend or an SSL checkpoint -- that architecture
   is covered separately by test_model_aasist_core.py.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch
import torch.nn as nn

from nkululeko.models.aasist_config import AasistConfig
from nkululeko.models.model_aasist import AasistModel, _WaveformDataset


def _default_cfg(**overrides):
    fields = {
        "device": "cpu",
        "ssl_model": "facebook/wav2vec2-xls-r-300m",
        "max_len": 16000,
        "batch_size": 2,
        "rawboost_algo": 0,
        "rawboost_n_f": 5,
        "rawboost_n_bands": 5,
        "rawboost_min_f": 20,
        "rawboost_max_f": 8000,
        "rawboost_min_bw": 100,
        "rawboost_max_bw": 1000,
        "rawboost_min_coeff": 10,
        "rawboost_max_coeff": 100,
        "rawboost_min_g": 0,
        "rawboost_max_g": 0,
        "rawboost_min_bias_lin_nonlin": 5,
        "rawboost_max_bias_lin_nonlin": 20,
        "rawboost_p": 10,
        "rawboost_g_sd": 2,
        "rawboost_snr_min": 10,
        "rawboost_snr_max": 40,
    }
    fields.update(overrides)
    return AasistConfig(**fields)


def _write_wav(path, seconds, sr=16000):
    rng = np.random.default_rng(0)
    signal = rng.uniform(-0.1, 0.1, size=int(seconds * sr)).astype(np.float32)
    sf.write(path, signal, sr)


class TestWaveformDataset:
    def test_short_clip_is_tiled_to_max_len(self, tmp_path):
        wav_path = tmp_path / "short.wav"
        _write_wav(wav_path, seconds=0.2)  # 3200 samples, shorter than max_len
        index = pd.MultiIndex.from_tuples(
            [(str(wav_path), pd.Timedelta(0), pd.NaT)],
            names=["file", "start", "end"],
        )
        df = pd.DataFrame({"label": [0]}, index=index)

        dataset = _WaveformDataset(
            df, target="label", cfg=_default_cfg(), augment=False
        )
        waveform, label = dataset[0]

        assert waveform.shape == (16000,)
        assert label == 0

    def test_long_clip_is_truncated_to_max_len(self, tmp_path):
        wav_path = tmp_path / "long.wav"
        _write_wav(wav_path, seconds=2.0)  # 32000 samples, longer than max_len
        index = pd.MultiIndex.from_tuples(
            [(str(wav_path), pd.Timedelta(0), pd.NaT)],
            names=["file", "start", "end"],
        )
        df = pd.DataFrame({"label": [1]}, index=index)

        dataset = _WaveformDataset(
            df, target="label", cfg=_default_cfg(), augment=False
        )
        waveform, label = dataset[0]

        assert waveform.shape == (16000,)
        assert label == 1

    def test_segmented_index_reads_only_the_segment(self, tmp_path):
        wav_path = tmp_path / "segmented.wav"
        _write_wav(wav_path, seconds=2.0, sr=16000)
        # A real segment (0.2s to 0.4s), not a whole-file NaT row.
        index = pd.MultiIndex.from_tuples(
            [(str(wav_path), pd.Timedelta(seconds=0.2), pd.Timedelta(seconds=0.4))],
            names=["file", "start", "end"],
        )
        df = pd.DataFrame({"label": [0]}, index=index)

        dataset = _WaveformDataset(
            df, target="label", cfg=_default_cfg(max_len=3200), augment=False
        )
        waveform, _ = dataset[0]

        # 0.2s at 16kHz = 3200 samples, matching max_len exactly (no tiling).
        assert waveform.shape == (3200,)

    def test_rawboost_applied_only_when_augment_true(self, tmp_path, monkeypatch):
        wav_path = tmp_path / "clip.wav"
        _write_wav(wav_path, seconds=0.5)
        index = pd.MultiIndex.from_tuples(
            [(str(wav_path), pd.Timedelta(0), pd.NaT)],
            names=["file", "start", "end"],
        )
        df = pd.DataFrame({"label": [0]}, index=index)
        cfg = _default_cfg(rawboost_algo=2)

        calls = []

        def fake_rawboost(signal, sr, cfg, algo):
            calls.append(algo)
            return signal

        monkeypatch.setattr(
            "nkululeko.models.model_aasist.apply_rawboost", fake_rawboost
        )

        train_ds = _WaveformDataset(df, target="label", cfg=cfg, augment=True)
        train_ds[0]
        assert calls == [2]

        eval_ds = _WaveformDataset(df, target="label", cfg=cfg, augment=False)
        eval_ds[0]
        assert calls == [2]  # unchanged -- not called again for augment=False


class _TinyNet(nn.Module):
    """Stand-in for AasistBackend: maps a raw waveform straight to 2
    logits via mean-pooling + a linear layer, so evaluate()/get_probas()
    can be tested without the real SSL frontend or graph-attention stack
    (covered separately by test_model_aasist_core.py)."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1, 2)

    def forward(self, x):
        pooled = x.mean(dim=1, keepdim=True)
        return self.fc(pooled)


@pytest.fixture
def aasist_model():
    df_train = pd.DataFrame({"label": [0, 1, 0, 1]})
    df_test = pd.DataFrame({"label": [1, 0]})

    with patch.object(AasistModel, "__init__", return_value=None):
        model = AasistModel(df_train, df_test, pd.DataFrame(), pd.DataFrame())
        model.target = "label"
        model.class_num = 2
        model.device = "cpu"
        model.run = 0
        model.epoch = 0
        model.df_test = df_test
        model.net = _TinyNet()
        model.criterion = nn.CrossEntropyLoss()
        model.context = type("Ctx", (), {"labels": ["real", "fake"]})()
        return model


class TestEvaluateAndProbas:
    def test_evaluate_returns_predictions_for_every_row(self, aasist_model):
        loader = [
            (torch.zeros(2, 10), torch.tensor([0, 1])),
        ]
        uar, targets, predictions, logits, loss_eval = aasist_model.evaluate(loader)

        assert len(targets) == 2
        assert len(predictions) == 2
        assert logits.shape == (2, 2)
        assert 0.0 <= uar <= 1.0
        assert loss_eval >= 0.0

    def test_get_probas_indexes_by_df_test_and_sums_to_one(self, aasist_model):
        logits = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
        probas = aasist_model.get_probas(logits)

        assert list(probas.index) == list(aasist_model.df_test.index)
        row_sums = probas.sum(axis=1).to_numpy()
        np.testing.assert_allclose(row_sums, [1.0, 1.0], rtol=1e-5)
