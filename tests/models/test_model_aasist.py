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
from nkululeko.optimizers.sam import SAM


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
        "domain_balanced_sampling": False,
        "ssl_layer_pooling": "last",
        "freeze_ssl_frontend": False,
        "dann_columns": [],
        "dann_lambda": 1.0,
        "dann_weight": 1.0,
        "dann_reverse": True,
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
    (covered separately by test_model_aasist_core.py). Also supports
    return_features=True (feat_dim=1, the pooled scalar itself) so
    DANN's train()-loop wiring can be tested the same way."""

    feat_dim = 1

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1, 2)

    def forward(self, x, return_features=False):
        pooled = x.mean(dim=1, keepdim=True)
        logits = self.fc(pooled)
        if return_features:
            return logits, pooled
        return logits


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


class TestGetLoaderDomainBalancedDispatch:
    """get_loader() must only reach for DomainBalancedBatchSampler on the
    training split (augment=True) when AASIST.domain_balanced_sampling is
    on -- never for dev/test, and never when the flag is off."""

    def _model_with_cfg(self, domain_balanced_sampling, n_jobs=0):
        with patch.object(AasistModel, "__init__", return_value=None):
            model = AasistModel(pd.DataFrame(), pd.DataFrame(), None, None)
            model.cfg = _default_cfg(domain_balanced_sampling=domain_balanced_sampling)
            model.target = "label"
            model.n_jobs = n_jobs
            model.dann_label_maps = {}
            return model

    def _df_with_domains(self):
        index = pd.MultiIndex.from_tuples(
            [(f"/f{i}.wav", pd.Timedelta(0), pd.NaT) for i in range(8)],
            names=["file", "start", "end"],
        )
        return pd.DataFrame(
            {"label": [0, 1] * 4, "source_db": (["a", "b"] * 4)}, index=index
        )

    def test_uses_batch_sampler_for_train_when_enabled(self):
        model = self._model_with_cfg(domain_balanced_sampling=True)
        loader = model.get_loader(self._df_with_domains(), augment=True, shuffle=True)
        from nkululeko.data.domain_sampler import DomainBalancedBatchSampler

        assert isinstance(loader.batch_sampler, DomainBalancedBatchSampler)

    def test_plain_loader_for_dev_test_even_when_enabled(self):
        model = self._model_with_cfg(domain_balanced_sampling=True)
        loader = model.get_loader(self._df_with_domains(), augment=False, shuffle=False)
        from nkululeko.data.domain_sampler import DomainBalancedBatchSampler

        assert not isinstance(loader.batch_sampler, DomainBalancedBatchSampler)

    def test_plain_loader_for_train_when_disabled(self):
        model = self._model_with_cfg(domain_balanced_sampling=False)
        loader = model.get_loader(self._df_with_domains(), augment=True, shuffle=True)
        from nkululeko.data.domain_sampler import DomainBalancedBatchSampler

        assert not isinstance(loader.batch_sampler, DomainBalancedBatchSampler)


class TestGetLoaderNumWorkers:
    """get_loader() must parallelize _WaveformDataset's per-item audio I/O
    (+ optional RawBoost) via MODEL.n_jobs -- a single-process loader
    (num_workers=0) serializes that CPU-bound work with GPU compute,
    which is why AASIST + RawBoost ran ~3x slower per epoch than bare
    AASIST. self.n_jobs is set by the base Model class from MODEL.n_jobs
    (default 8); 0 means "no extra workers", matching DataLoader's own
    default and this project's other CPU-light models (e.g. ADM's
    TensorDataset-backed loader, which has no per-item work to
    parallelize).
    """

    def _model_with_cfg(self, n_jobs):
        with patch.object(AasistModel, "__init__", return_value=None):
            model = AasistModel(pd.DataFrame(), pd.DataFrame(), None, None)
            model.cfg = _default_cfg()
            model.target = "label"
            model.n_jobs = n_jobs
            model.dann_label_maps = {}
            return model

    def _df(self):
        index = pd.MultiIndex.from_tuples(
            [(f"/f{i}.wav", pd.Timedelta(0), pd.NaT) for i in range(4)],
            names=["file", "start", "end"],
        )
        return pd.DataFrame({"label": [0, 1, 0, 1]}, index=index)

    def test_positive_n_jobs_sets_num_workers(self):
        model = self._model_with_cfg(n_jobs=4)
        loader = model.get_loader(self._df(), augment=False, shuffle=False)
        assert loader.num_workers == 4
        assert loader.persistent_workers is True

    def test_zero_n_jobs_keeps_single_process_loader(self):
        model = self._model_with_cfg(n_jobs=0)
        loader = model.get_loader(self._df(), augment=False, shuffle=False)
        assert loader.num_workers == 0

    def test_num_workers_applied_to_domain_balanced_loader_too(self):
        with patch.object(AasistModel, "__init__", return_value=None):
            model = AasistModel(pd.DataFrame(), pd.DataFrame(), None, None)
            model.cfg = _default_cfg(domain_balanced_sampling=True)
            model.target = "label"
            model.n_jobs = 2
            model.dann_label_maps = {}

        index = pd.MultiIndex.from_tuples(
            [(f"/f{i}.wav", pd.Timedelta(0), pd.NaT) for i in range(8)],
            names=["file", "start", "end"],
        )
        df = pd.DataFrame(
            {"label": [0, 1] * 4, "source_db": (["a", "b"] * 4)}, index=index
        )
        loader = model.get_loader(df, augment=True, shuffle=True)
        assert loader.num_workers == 2


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


class TestTrainSamBranch:
    """train() must dispatch to the SAM closure path (two forward passes
    per batch) only when MODEL.sam wrapped self.optimizer in SAM -- and
    must still work identically to before when it didn't. Mirrors the
    ADMModel.train() SAM wiring (same is_sam_optimizer() check, same
    closure shape) -- one mechanism, two independent model types."""

    def _trainable_model(self, optimizer_factory):
        df_train = pd.DataFrame({"label": [0, 1, 0, 1]})
        df_test = pd.DataFrame({"label": [1, 0]})
        with patch.object(AasistModel, "__init__", return_value=None):
            model = AasistModel(df_train, df_test, pd.DataFrame(), pd.DataFrame())
            model.target = "label"
            model.class_num = 2
            model.device = "cpu"
            model.net = _TinyNet()
            model.criterion = nn.CrossEntropyLoss()
            model.cfg = _default_cfg()
            model.dann_heads = None
            model.optimizer = optimizer_factory(model.net.parameters())
            model.scheduler = None
            model.scheduler_type = "none"
            model.scheduler_needs_init = False
            model.trainloader = [
                (torch.randn(4, 16000), torch.tensor([0, 1, 0, 1])),
                (torch.randn(4, 16000), torch.tensor([1, 0, 1, 0])),
            ]
            return model

    def test_plain_optimizer_trains_one_epoch(self):
        model = self._trainable_model(lambda params: torch.optim.SGD(params, lr=0.01))
        before = model.net.fc.weight.clone()

        model.train()

        assert hasattr(model, "loss")
        assert torch.isfinite(torch.tensor(model.loss))
        assert not torch.allclose(model.net.fc.weight, before)

    def test_sam_optimizer_trains_one_epoch_via_closure(self):
        model = self._trainable_model(
            lambda params: SAM(params, torch.optim.SGD, rho=0.05, lr=0.01)
        )
        before = model.net.fc.weight.clone()

        model.train()

        assert hasattr(model, "loss")
        assert torch.isfinite(torch.tensor(model.loss))
        # SAM's second_step() restores pre-ascent weights before the base
        # optimizer's real update -- net effect must still be a real
        # step away from the starting weights, not a no-op and not left
        # sitting at the ascent-perturbed point.
        assert not torch.allclose(model.net.fc.weight, before)


class TestWaveformDatasetDann:
    """_WaveformDataset must emit the 3-tuple (waveform, label,
    domain_labels) form only on the training split (augment=True) with
    DANN enabled (cfg.dann_columns non-empty) -- dev/test and the
    DANN-off default must stay byte-identical 2-tuples."""

    def _df_with_domain(self, tmp_path, n=4):
        paths = []
        for i in range(n):
            p = tmp_path / f"f{i}.wav"
            _write_wav(p, seconds=0.5)
            paths.append(str(p))
        index = pd.MultiIndex.from_tuples(
            [(p, pd.Timedelta(0), pd.NaT) for p in paths],
            names=["file", "start", "end"],
        )
        return pd.DataFrame(
            {"label": [0, 1] * (n // 2), "source_db": (["a", "b"] * (n // 2))},
            index=index,
        )

    def test_two_tuple_when_dann_off(self, tmp_path):
        cfg = _default_cfg(dann_columns=[])
        df = self._df_with_domain(tmp_path)
        ds = _WaveformDataset(df, "label", cfg, augment=True, dann_label_maps={})

        item = ds[0]

        assert len(item) == 2

    def test_three_tuple_when_train_and_dann_on(self, tmp_path):
        cfg = _default_cfg(dann_columns=["source_db"])
        df = self._df_with_domain(tmp_path)
        label_maps = {"source_db": {"a": 0, "b": 1}}
        ds = _WaveformDataset(
            df, "label", cfg, augment=True, dann_label_maps=label_maps
        )

        waveform, label, domain_labels = ds[0]

        assert domain_labels.shape == (1,)
        assert domain_labels.dtype == torch.long
        assert domain_labels[0].item() == label_maps["source_db"][df["source_db"].iloc[0]]

    def test_two_tuple_for_dev_test_even_when_dann_columns_set(self, tmp_path):
        """augment=False (dev/test split) must never emit domain labels,
        even when cfg.dann_columns is non-empty -- DANN only ever trains
        on the training split."""
        cfg = _default_cfg(dann_columns=["source_db"])
        df = self._df_with_domain(tmp_path)
        label_maps = {"source_db": {"a": 0, "b": 1}}
        ds = _WaveformDataset(
            df, "label", cfg, augment=False, dann_label_maps=label_maps
        )

        item = ds[0]

        assert len(item) == 2


class TestBuildDannHeads:
    """AasistModel._build_dann_heads() builds one DomainAdversarialHead
    per MODEL.dann_columns entry from df_train's own values."""

    def _model(self):
        with patch.object(AasistModel, "__init__", return_value=None):
            model = AasistModel(pd.DataFrame(), pd.DataFrame(), None, None)
            model.net = _TinyNet()
            model.device = "cpu"
            return model

    def test_no_columns_leaves_heads_none(self):
        model = self._model()
        model.cfg = _default_cfg(dann_columns=[])
        model.util = type("U", (), {"debug": lambda self, m: None})()

        model._build_dann_heads(pd.DataFrame({"label": [0, 1]}))

        assert model.dann_heads is None
        assert model.dann_label_maps == {}

    def test_builds_one_head_per_column_with_correct_class_count(self):
        model = self._model()
        model.cfg = _default_cfg(dann_columns=["source_db"])
        model.util = type("U", (), {"debug": lambda self, m: None})()
        df_train = pd.DataFrame(
            {"label": [0, 1, 0, 1], "source_db": ["a", "b", "c", "a"]}
        )

        model._build_dann_heads(df_train)

        assert set(model.dann_heads.keys()) == {"source_db"}
        assert model.dann_label_maps["source_db"] == {"a": 0, "b": 1, "c": 2}
        assert model.dann_heads["source_db"].classifier[-1].out_features == 3
        assert model.dann_heads["source_db"].classifier[0].in_features == model.net.feat_dim

    def test_errors_on_fewer_than_two_unique_values(self):
        model = self._model()
        model.cfg = _default_cfg(dann_columns=["source_db"])

        def _raise_error(self, msg):
            raise ValueError(msg)

        model.util = type("U", (), {"error": _raise_error, "debug": lambda self, m: None})()
        df_train = pd.DataFrame({"label": [0, 1], "source_db": ["a", "a"]})

        with pytest.raises(ValueError, match="dann_columns"):
            model._build_dann_heads(df_train)


class TestTrainDannBranch:
    """train() must dispatch to the DANN combined-loss path (main task
    loss + per-column adversarial loss) only when self.dann_heads is
    set, unpacking the trainloader's 3-tuple batches -- and the DANN
    head's own parameters must actually receive gradient updates (proof
    the optimizer was built over both self.net and self.dann_heads)."""

    def _model_with_dann(self):
        df_train = pd.DataFrame({"label": [0, 1, 0, 1]})
        df_test = pd.DataFrame({"label": [1, 0]})
        with patch.object(AasistModel, "__init__", return_value=None):
            model = AasistModel(df_train, df_test, pd.DataFrame(), pd.DataFrame())
            model.target = "label"
            model.class_num = 2
            model.device = "cpu"
            model.net = _TinyNet()
            model.criterion = nn.CrossEntropyLoss()
            model.cfg = _default_cfg(dann_columns=["source_db"], dann_weight=1.0)
            model.util = type("U", (), {"debug": lambda self, m: None})()
            model._build_dann_heads(
                pd.DataFrame({"label": [0, 1, 0, 1], "source_db": ["a", "b", "a", "b"]})
            )
            import itertools

            model.optimizer = torch.optim.SGD(
                itertools.chain(model.net.parameters(), model.dann_heads.parameters()),
                lr=0.05,
            )
            model.scheduler = None
            model.scheduler_type = "none"
            model.scheduler_needs_init = False
            model.trainloader = [
                (
                    torch.randn(4, 16000),
                    torch.tensor([0, 1, 0, 1]),
                    torch.tensor([[0], [1], [0], [1]]),
                ),
                (
                    torch.randn(4, 16000),
                    torch.tensor([1, 0, 1, 0]),
                    torch.tensor([[1], [0], [1], [0]]),
                ),
            ]
            return model

    def test_train_runs_and_updates_both_net_and_dann_head(self):
        model = self._model_with_dann()
        net_before = model.net.fc.weight.clone()
        head_before = model.dann_heads["source_db"].classifier[0].weight.clone()

        model.train()

        assert hasattr(model, "loss")
        assert torch.isfinite(torch.tensor(model.loss))
        assert not torch.allclose(model.net.fc.weight, net_before)
        assert not torch.allclose(
            model.dann_heads["source_db"].classifier[0].weight, head_before
        )
