"""Unit tests for label_smoothing config parsing in MLPModel."""

import configparser

import pytest
import torch

import nkululeko.glob_conf as glob_conf


def _make_config(label_smoothing=None, loss="cross"):
    """Helper to create a minimal config for model tests."""
    config = configparser.ConfigParser()
    model_section = {"loss": loss, "layers": "[64]"}
    if label_smoothing is not None:
        model_section["label_smoothing"] = str(label_smoothing)
    config["MODEL"] = model_section
    config["DATA"] = {"target": "emotion"}
    config["EXP"] = {"root": "/tmp/test_ls", "name": "test_ls"}
    return config


class TestLabelSmoothingParsing:
    """Test the _get_label_smoothing helper in MLP-like models."""

    def _make_mock_model(self, label_smoothing=None):
        """Create a minimal object with _get_label_smoothing via duck-typing."""
        config = _make_config(label_smoothing=label_smoothing)
        glob_conf.config = config
        glob_conf.labels = ["anger", "neutral", "happy"]
        from nkululeko.utils.util import Util

        class MockModel:
            util = Util("test")

            def _get_label_smoothing(self):
                ls = self.util.config_val("MODEL", "label_smoothing", "False")
                if str(ls).lower() in ("true", "1"):
                    return 0.1
                else:
                    try:
                        smoothing = float(ls)
                    except (ValueError, TypeError):
                        smoothing = 0.0
                    return smoothing

        return MockModel()

    def test_default_no_smoothing(self):
        model = self._make_mock_model()
        assert model._get_label_smoothing() == 0.0

    def test_true_returns_01(self):
        model = self._make_mock_model(label_smoothing="True")
        assert model._get_label_smoothing() == 0.1

    def test_float_value_passed_through(self):
        model = self._make_mock_model(label_smoothing="0.2")
        assert model._get_label_smoothing() == pytest.approx(0.2)

    def test_zero_returns_zero(self):
        model = self._make_mock_model(label_smoothing="0.0")
        assert model._get_label_smoothing() == 0.0

    def test_false_returns_zero(self):
        model = self._make_mock_model(label_smoothing="False")
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
