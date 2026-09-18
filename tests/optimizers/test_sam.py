"""Unit tests for the model-agnostic SAM optimizer wrapper
(nkululeko/optimizers/sam.py) and its MODEL.sam integration in
get_optimizer()."""

import pytest
import torch
import torch.nn as nn

from nkululeko.optimizers.optimizer_factory import get_optimizer
from nkululeko.optimizers.sam import SAM, is_sam_optimizer


class MockUtil:
    def __init__(self, config_dict=None):
        self.config = config_dict or {}
        self.debug_messages = []

    def config_val(self, section, key, default):
        return self.config.get(f"{section}.{key}", default)

    def config_val_bool(self, section, key, default=False):
        val = self.config_val(section, key, str(default))
        return str(val).strip().lower() in ("true", "1", "yes")

    def debug(self, message):
        self.debug_messages.append(message)

    def error(self, message):
        raise ValueError(message)


@pytest.fixture
def simple_model():
    torch.manual_seed(0)
    return nn.Linear(10, 5)


class TestSAMConstruction:
    def test_rejects_negative_rho(self, simple_model):
        with pytest.raises(ValueError, match="rho"):
            SAM(simple_model.parameters(), torch.optim.Adam, rho=-0.1)

    def test_wraps_base_optimizer_params(self, simple_model):
        sam = SAM(simple_model.parameters(), torch.optim.Adam, rho=0.05, lr=0.001)
        assert isinstance(sam.base_optimizer, torch.optim.Adam)
        assert sam.param_groups[0]["lr"] == pytest.approx(0.001)


class TestSAMStep:
    def test_step_requires_closure(self, simple_model):
        sam = SAM(simple_model.parameters(), torch.optim.Adam, rho=0.05, lr=0.001)
        with pytest.raises(ValueError, match="closure"):
            sam.step(None)

    def test_step_restores_weights_before_base_update(self, simple_model):
        """After step(), parameters must reflect the base optimizer's
        update from the *original* weights, not be left at the
        ascent-perturbed point -- second_step() restores old_p before
        calling base_optimizer.step()."""
        sam = SAM(simple_model.parameters(), torch.optim.SGD, rho=0.05, lr=1.0)
        original = simple_model.weight.data.clone()

        call_count = {"n": 0}

        def closure():
            call_count["n"] += 1
            sam.zero_grad()
            x = torch.randn(4, 10)
            loss = simple_model(x).sum()
            loss.backward()
            return loss

        sam.step(closure)

        assert call_count["n"] == 2  # once at current weights, once at ascent point
        # Plain SGD with lr=1.0 subtracts grad*lr from the *original*
        # weights (post second_step restore) -- so the update should be
        # a real step away from `original`, not a no-op and not still at
        # the perturbed point.
        assert not torch.allclose(simple_model.weight.data, original)

    def test_ascent_step_perturbs_in_gradient_direction(self, simple_model):
        sam = SAM(simple_model.parameters(), torch.optim.SGD, rho=0.5, lr=0.0)
        x = torch.randn(4, 10)
        loss = simple_model(x).sum()
        loss.backward()

        before = simple_model.weight.data.clone()
        grad = simple_model.weight.grad.clone()
        sam.first_step()
        after = simple_model.weight.data

        # e_w = rho * grad / ||grad|| -- perturbation must point along
        # +grad (ascent, toward higher loss), not away from it.
        delta = after - before
        assert torch.allclose(torch.sign(delta), torch.sign(grad), equal_nan=True)

    def test_second_step_restores_pre_ascent_weights_before_update(self, simple_model):
        sam = SAM(simple_model.parameters(), torch.optim.SGD, rho=0.5, lr=0.0)
        x = torch.randn(4, 10)
        loss = simple_model(x).sum()
        loss.backward()
        original = simple_model.weight.data.clone()

        sam.first_step()
        assert not torch.allclose(simple_model.weight.data, original)

        # lr=0.0 means the base optimizer's own update is a no-op, so
        # after second_step() the weights must be exactly the
        # pre-ascent originals again.
        sam.second_step()
        assert torch.allclose(simple_model.weight.data, original)


class TestIsSamOptimizer:
    def test_true_for_sam_instance(self, simple_model):
        sam = SAM(simple_model.parameters(), torch.optim.Adam, rho=0.05, lr=0.001)
        assert is_sam_optimizer(sam) is True

    def test_false_for_plain_optimizer(self, simple_model):
        plain = torch.optim.Adam(simple_model.parameters(), lr=0.001)
        assert is_sam_optimizer(plain) is False


class TestGetOptimizerSamIntegration:
    def test_sam_off_by_default_returns_plain_optimizer(self, simple_model):
        util = MockUtil({})
        optimizer, _ = get_optimizer(simple_model.parameters(), util)
        assert not is_sam_optimizer(optimizer)

    def test_sam_true_wraps_adamw(self, simple_model):
        util = MockUtil({"MODEL.sam": "True", "MODEL.optimizer": "adamw"})
        optimizer, lr = get_optimizer(simple_model.parameters(), util, default_lr=0.001)

        assert is_sam_optimizer(optimizer)
        assert isinstance(optimizer.base_optimizer, torch.optim.AdamW)
        assert lr == pytest.approx(0.001)
        assert any("sam" in msg.lower() for msg in util.debug_messages)

    def test_sam_true_wraps_sgd_with_momentum(self, simple_model):
        util = MockUtil(
            {"MODEL.sam": "True", "MODEL.optimizer": "sgd", "MODEL.momentum": "0.9"}
        )
        optimizer, _ = get_optimizer(simple_model.parameters(), util)

        assert is_sam_optimizer(optimizer)
        assert isinstance(optimizer.base_optimizer, torch.optim.SGD)
        assert optimizer.param_groups[0]["momentum"] == pytest.approx(0.9)

    def test_sam_rho_overridable(self, simple_model):
        util = MockUtil({"MODEL.sam": "True", "MODEL.sam_rho": "0.1"})
        optimizer, _ = get_optimizer(simple_model.parameters(), util)

        assert optimizer.param_groups[0]["rho"] == pytest.approx(0.1)

    def test_sam_rho_default(self, simple_model):
        util = MockUtil({"MODEL.sam": "True"})
        optimizer, _ = get_optimizer(simple_model.parameters(), util)

        assert optimizer.param_groups[0]["rho"] == pytest.approx(0.05)

    def test_end_to_end_step_with_get_optimizer(self, simple_model):
        """The SAM instance get_optimizer() returns must actually be
        usable via the closure-based step() -- not just constructible."""
        util = MockUtil({"MODEL.sam": "True", "MODEL.optimizer": "sgd"})
        optimizer, _ = get_optimizer(simple_model.parameters(), util, default_lr=0.01)

        def closure():
            optimizer.zero_grad()
            loss = simple_model(torch.randn(4, 10)).sum()
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        assert torch.isfinite(loss)
