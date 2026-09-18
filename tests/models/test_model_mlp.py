import numpy as np
import pandas as pd
import pytest
import torch
from unittest.mock import patch

from nkululeko.models.model_mlp import MLPModel


class DummyUtil:
    def config_val(self, section, key, default=None):
        # Provide defaults for required config values
        if key == "manual_seed":
            return True
        if key == "loss":
            return "cross"
        if key == "device":
            return "cpu"
        if key == "learning_rate":
            return 0.001
        if key == "batch_size":
            return 2
        if key == "drop":
            return False
        return default

    def debug(self, msg):
        pass

    def error(self, msg):
        raise Exception(msg)

    def get_path(self, key):
        return "./"

    def get_exp_name(self, only_train=False):
        return "exp"


@pytest.fixture(autouse=True)
def patch_globals(monkeypatch):
    # Patch global config and labels
    import nkululeko.glob_conf as glob_conf

    glob_conf.config = {
        "DATA": {"target": "label"},
        "MODEL": {"layers": "{'a': 8, 'b': 4}"},
    }
    glob_conf.labels = [0, 1]
    yield


@pytest.fixture
def dummy_data():
    # 4 samples, 3 features
    feats_train = pd.DataFrame(np.random.rand(4, 3), columns=["f1", "f2", "f3"])
    feats_test = pd.DataFrame(np.random.rand(2, 3), columns=["f1", "f2", "f3"])
    df_train = pd.DataFrame({"label": [0, 1, 0, 1]})
    df_test = pd.DataFrame({"label": [1, 0]})
    return df_train, df_test, feats_train, feats_test


@pytest.fixture
def mlp_model(dummy_data, monkeypatch):
    df_train, df_test, feats_train, feats_test = dummy_data
    with patch.object(MLPModel, "__init__", return_value=None):
        model = MLPModel(df_train, df_test, feats_train, feats_test)
        model.util = DummyUtil()
        model.n_jobs = 1
        model.target = "label"
        model.class_num = 2
        model.criterion = torch.nn.CrossEntropyLoss()
        model.device = "cpu"
        model.learning_rate = 0.001
        model.batch_size = 2
        model.num_workers = 1
        model.domain_balanced_sampling = False
        model.loss = 0.0
        model.loss_eval = 0.0
        model.run = 0
        model.epoch = 0
        model.df_test = df_test
        model.feats_test = feats_test
        model.feats_train = feats_train

        # Create a simple MLP model for testing
        model.model = MLPModel.MLP(3, {"a": 8, "b": 4}, 2, False, torch.nn.ReLU()).to(
            "cpu"
        )
        model.optimizer = torch.optim.Adam(model.model.parameters(), lr=0.001)

        # Create data loaders
        model.trainloader = model.get_loader(feats_train, df_train, True)
        model.testloader = model.get_loader(feats_test, df_test, False)
        model.store_path = "/tmp/test_model.pt"

        return model


def test_mlpmodel_init(mlp_model):
    assert hasattr(mlp_model, "model")
    assert hasattr(mlp_model, "trainloader")
    assert hasattr(mlp_model, "testloader")
    assert mlp_model.model is not None


def test_train_and_predict(mlp_model):
    mlp_model.train()
    report = mlp_model.predict()
    assert hasattr(report, "result")
    assert hasattr(report.result, "train")


def test_get_predictions(mlp_model):
    mlp_model.train()
    preds, probas = mlp_model.get_predictions()
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == 2


def test_get_probas(mlp_model):
    mlp_model.train()
    _, _, _, logits = mlp_model.evaluate(
        mlp_model.model, mlp_model.testloader, mlp_model.device
    )
    probas = mlp_model.get_probas(logits)
    assert isinstance(probas, pd.DataFrame)
    assert set(probas.columns) == set([0, 1])


def test_predict_sample(mlp_model):
    mlp_model.train()
    feats = np.random.rand(3)
    res = mlp_model.predict_sample(feats)
    assert isinstance(res, dict)
    assert set(res.keys()) == set([0, 1])


def test_predict_shap(mlp_model):
    mlp_model.train()
    feats = pd.DataFrame(np.random.rand(2, 3))
    results = mlp_model.predict_shap(feats)
    assert len(results) == 2


def test_store_and_load(tmp_path, mlp_model, monkeypatch):
    mlp_model.train()

    # Mock the util methods that load() uses to construct the path
    def mock_get_path(key):
        if key == "model_dir":
            return str(tmp_path) + "/"
        return "./"

    def mock_get_exp_name(only_train=False):
        return "model"

    mlp_model.util.get_path = mock_get_path
    mlp_model.util.get_exp_name = mock_get_exp_name

    # Set store path to match what load() will construct
    mlp_model.store_path = str(tmp_path) + "/model_0_000.model"
    mlp_model.store()

    # Simulate loading
    mlp_model.load(0, 0)
    assert mlp_model.model is not None


def test_set_testdata(mlp_model, dummy_data):
    _, df_test, _, feats_test = dummy_data
    mlp_model.set_testdata(df_test, feats_test)
    assert mlp_model.testloader is not None


def test_reset_test(mlp_model, dummy_data):
    _, df_test, _, feats_test = dummy_data
    mlp_model.reset_test(df_test, feats_test)
    assert mlp_model.testloader is not None


def test_mlp_model_init_with_dropout_list():
    # Test with a list of dropout values
    mlp_inner = MLPModel.MLP(3, {"a": 8, "b": 4}, 2, [0.1, 0.2], torch.nn.ReLU())
    dropout_layers = [l for l in mlp_inner.linear if isinstance(l, torch.nn.Dropout)]
    assert len(dropout_layers) == 1
    assert dropout_layers[0].p == pytest.approx(0.1)


def test_mlp_model_init_with_dropout_float():
    # Test with a single float value for dropout
    mlp_inner = MLPModel.MLP(3, {"a": 8, "b": 4}, 2, 0.5, torch.nn.ReLU())
    dropout_layers = [l for l in mlp_inner.linear if isinstance(l, torch.nn.Dropout)]
    assert len(dropout_layers) == 1
    assert dropout_layers[0].p == pytest.approx(0.5)


def test_train_one_epoch_with_sam_optimizer(mlp_model):
    """train() must dispatch to the SAM closure path (two forward/backward
    passes per batch) when self.optimizer is SAM-wrapped -- mirrors
    AasistModel's and ADMModel's identical is_sam_optimizer() branch
    (MLPModel is the third model type this session wires SAM into from
    the same nkululeko.optimizers.sam implementation)."""
    from nkululeko.optimizers.sam import SAM

    mlp_model.optimizer = SAM(
        mlp_model.model.parameters(), torch.optim.SGD, rho=0.05, lr=0.01
    )
    before = next(mlp_model.model.parameters()).clone()

    mlp_model.train()

    assert mlp_model.loss is not None
    assert not torch.allclose(next(mlp_model.model.parameters()), before)


class TestGetLoaderDomainBalancedDispatch:
    """get_loader() must only reach for DomainBalancedBatchSampler on the
    training split (shuffle=True) when MODEL.domain_balanced_sampling is
    on -- never for dev/test, never when the flag is off. Mirrors
    AasistModel's and ADMModel's identical dispatch tests."""

    def _model_with(self, domain_balanced_sampling):
        with patch.object(MLPModel, "__init__", return_value=None):
            model = MLPModel(pd.DataFrame(), pd.DataFrame(), None, None)
            model.target = "label"
            model.batch_size = 2
            model.domain_balanced_sampling = domain_balanced_sampling
            return model

    def _feats_and_labels(self):
        feats = pd.DataFrame(np.random.rand(8, 3), columns=["f1", "f2", "f3"])
        labels = pd.DataFrame({"label": [0, 1] * 4, "source_db": (["a", "b"] * 4)})
        return feats, labels

    def test_uses_batch_sampler_for_train_when_enabled(self):
        model = self._model_with(domain_balanced_sampling=True)
        feats, labels = self._feats_and_labels()
        loader = model.get_loader(feats, labels, shuffle=True)
        from nkululeko.data.domain_sampler import DomainBalancedBatchSampler

        assert isinstance(loader.batch_sampler, DomainBalancedBatchSampler)

    def test_plain_loader_for_dev_test_even_when_enabled(self):
        model = self._model_with(domain_balanced_sampling=True)
        feats, labels = self._feats_and_labels()
        loader = model.get_loader(feats, labels, shuffle=False)
        from nkululeko.data.domain_sampler import DomainBalancedBatchSampler

        assert not isinstance(loader.batch_sampler, DomainBalancedBatchSampler)

    def test_plain_loader_for_train_when_disabled(self):
        model = self._model_with(domain_balanced_sampling=False)
        feats, labels = self._feats_and_labels()
        loader = model.get_loader(feats, labels, shuffle=True)
        from nkululeko.data.domain_sampler import DomainBalancedBatchSampler

        assert not isinstance(loader.batch_sampler, DomainBalancedBatchSampler)


def test_evaluate_works_with_batch_sampler_loader(mlp_model, dummy_data):
    """Regression test (pre-emptive here, reactive for ADM): evaluate()
    used to index output tensors via `index * loader.batch_size`, which
    is None for a loader built with batch_sampler= (domain-balanced
    sampling) -- fixed to a running offset before this ever shipped
    broken, unlike ADMModel's evaluate() which crashed in a live GPU run
    first."""
    df_train, df_test, feats_train, feats_test = dummy_data
    df_train = df_train.assign(source_db=["a", "b"] * 2)
    mlp_model.domain_balanced_sampling = True
    mlp_model.trainloader = mlp_model.get_loader(feats_train, df_train, shuffle=True)
    assert mlp_model.trainloader.batch_size is None  # batch_sampler in use

    uar, targets, predictions, logits = mlp_model.evaluate(
        mlp_model.model, mlp_model.trainloader, mlp_model.device
    )

    assert len(targets) == 4
    assert len(predictions) == 4
    assert 0.0 <= uar <= 1.0
