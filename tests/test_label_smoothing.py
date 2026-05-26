"""Unit tests for label_smoothing config parsing in the base Model class."""

import configparser
import tempfile

import pytest
import torch

import nkululeko.glob_conf as glob_conf
from nkululeko.models.model import Model
from nkululeko.utils.util import Util


def _make_config(label_smoothing=None, loss="cross"):
    """Helper to create a minimal config for model tests."""
    config = configparser.ConfigParser()
    model_section = {"loss": loss, "layers": "[64]"}
    if label_smoothing is not None:
        model_section["label_smoothing"] = str(label_smoothing)
    config["MODEL"] = model_section
    config["DATA"] = {"target": "emotion"}
    config["EXP"] = {"root": tempfile.gettempdir(), "name": "test_ls"}
    return config


def _make_model_stub(label_smoothing=None):
    """Return a Model instance with only the attributes needed by _get_label_smoothing.

    Uses ``object.__new__`` to bypass ``Model.__init__``, which requires
    train/test dataframes that are unnecessary for this unit test.
    """
    config = _make_config(label_smoothing=label_smoothing)
    glob_conf.config = config
    glob_conf.labels = ["anger", "neutral", "happy"]
    model = object.__new__(Model)
    model.util = Util("test")
    return model


class TestLabelSmoothingParsing:
    """Test the _get_label_smoothing helper defined on the base Model class."""

    def test_default_no_smoothing(self):
        model = _make_model_stub()
        assert model._get_label_smoothing() == 0.0

    def test_true_returns_01(self):
        model = _make_model_stub(label_smoothing="True")
        assert model._get_label_smoothing() == 0.1

    def test_float_value_passed_through(self):
        model = _make_model_stub(label_smoothing="0.2")
        assert model._get_label_smoothing() == pytest.approx(0.2)

    def test_zero_returns_zero(self):
        model = _make_model_stub(label_smoothing="0.0")
        assert model._get_label_smoothing() == 0.0

    def test_false_returns_zero(self):
        model = _make_model_stub(label_smoothing="False")
        assert model._get_label_smoothing() == 0.0


class TestCrossEntropyWithLabelSmoothing:
    """Verify that torch.nn.CrossEntropyLoss uses label_smoothing correctly."""

    def test_no_smoothing_default(self):
        loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.0)
        logits = torch.tensor([[10.0, 0.0, 0.0]])
        target = torch.tensor([0])
        loss = loss_fn(logits, target)
        assert loss.item() < 0.01

    def test_smoothing_increases_loss_on_confident_correct(self):
        """Label smoothing should increase loss for over-confident correct predictions."""
        logits = torch.tensor([[10.0, 0.0, 0.0]])
        target = torch.tensor([0])
        loss_no_smooth = torch.nn.CrossEntropyLoss(label_smoothing=0.0)(logits, target)
        loss_smooth = torch.nn.CrossEntropyLoss(label_smoothing=0.1)(logits, target)
        assert loss_smooth.item() > loss_no_smooth.item()

    def test_smoothing_on_confident_incorrect_prediction(self):
        """Label smoothing should decrease loss for highly confident incorrect predictions."""
        logits = torch.tensor([[0.0, 10.0, 0.0]])
        target = torch.tensor([0])
        loss_no_smooth = torch.nn.CrossEntropyLoss(label_smoothing=0.0)(logits, target)
        loss_smooth = torch.nn.CrossEntropyLoss(label_smoothing=0.1)(logits, target)
        assert loss_smooth.item() < loss_no_smooth.item()
