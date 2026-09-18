"""Unit tests for the model-agnostic DANN components
(nkululeko/models/domain_adversarial.py)."""

import math

import pytest
import torch
import torch.nn as nn

from nkululeko.models.domain_adversarial import (
    DomainAdversarialHead,
    GradientReversalLayer,
    grl_lambda_schedule,
)


class TestGradientReversalLayer:
    def test_forward_is_identity(self):
        grl = GradientReversalLayer(lambda_=1.0)
        x = torch.randn(4, 8)
        assert torch.equal(grl(x), x)

    def test_backward_negates_and_scales_gradient(self):
        grl = GradientReversalLayer(lambda_=2.0)
        x = torch.randn(4, 8, requires_grad=True)
        y = grl(x)
        y.sum().backward()

        # d(sum(y))/dx would be all-ones without reversal; with
        # lambda_=2.0 it must be all -2.0.
        assert torch.allclose(x.grad, torch.full_like(x, -2.0))

    def test_lambda_is_mutable_between_forward_calls(self):
        grl = GradientReversalLayer(lambda_=1.0)
        x1 = torch.randn(2, 4, requires_grad=True)
        grl(x1).sum().backward()
        assert torch.allclose(x1.grad, torch.full_like(x1, -1.0))

        grl.lambda_ = 0.5
        x2 = torch.randn(2, 4, requires_grad=True)
        grl(x2).sum().backward()
        assert torch.allclose(x2.grad, torch.full_like(x2, -0.5))


class TestGrlLambdaSchedule:
    def test_zero_progress_is_zero(self):
        assert grl_lambda_schedule(0.0) == pytest.approx(0.0)

    def test_full_progress_approaches_one(self):
        assert grl_lambda_schedule(1.0) == pytest.approx(1.0, abs=1e-3)

    def test_monotonically_increasing(self):
        xs = [i / 10 for i in range(11)]
        ys = [grl_lambda_schedule(x) for x in xs]
        assert all(b >= a for a, b in zip(ys, ys[1:]))

    def test_clamps_out_of_range_progress(self):
        assert grl_lambda_schedule(-1.0) == pytest.approx(grl_lambda_schedule(0.0))
        assert grl_lambda_schedule(2.0) == pytest.approx(grl_lambda_schedule(1.0))

    def test_gamma_controls_steepness(self):
        # A larger gamma ramps faster -- at progress=0.5, a bigger gamma
        # must be closer to 1.0 (already saturated) than a smaller one.
        slow = grl_lambda_schedule(0.5, gamma=1.0)
        fast = grl_lambda_schedule(0.5, gamma=20.0)
        assert fast > slow


class TestDomainAdversarialHead:
    def test_output_shape(self):
        head = DomainAdversarialHead(feat_dim=16, num_classes=4)
        feats = torch.randn(3, 16)
        logits = head(feats)
        assert logits.shape == (3, 4)

    def test_reverse_true_negates_upstream_gradient(self):
        head = DomainAdversarialHead(feat_dim=8, num_classes=2, reverse=True, lambda_=1.0)
        feats = torch.randn(4, 8, requires_grad=True)
        logits = head(feats)
        loss = logits.sum()
        loss.backward()

        # Reversal flips the sign of d(loss)/d(feats) relative to what a
        # plain (non-reversed) head with identical weights would produce.
        plain_head = DomainAdversarialHead(
            feat_dim=8, num_classes=2, reverse=False
        )
        plain_head.classifier.load_state_dict(head.classifier.state_dict())
        feats2 = feats.detach().clone().requires_grad_(True)
        plain_head(feats2).sum().backward()

        assert torch.allclose(feats.grad, -feats2.grad, atol=1e-6)

    def test_reverse_false_is_plain_multitask_head(self):
        head = DomainAdversarialHead(feat_dim=8, num_classes=3, reverse=False)
        assert isinstance(head.grl, nn.Identity)

        feats = torch.randn(2, 8, requires_grad=True)
        head(feats).sum().backward()  # must run without error, no reversal involved

        # lambda_ has no meaning without a GRL -- setter is a no-op, getter is fixed at 1.0.
        head.lambda_ = 99.0
        assert head.lambda_ == 1.0

    def test_lambda_property_reads_through_to_grl(self):
        head = DomainAdversarialHead(feat_dim=4, num_classes=2, reverse=True, lambda_=0.3)
        assert head.lambda_ == pytest.approx(0.3)

        head.lambda_ = 0.7
        assert head.grl.lambda_ == pytest.approx(0.7)

    def test_gradient_scales_with_lambda(self):
        head = DomainAdversarialHead(feat_dim=6, num_classes=2, reverse=True, lambda_=2.0)
        feats = torch.randn(3, 6, requires_grad=True)
        head(feats).sum().backward()
        grad_at_2 = feats.grad.clone()

        head2 = DomainAdversarialHead(feat_dim=6, num_classes=2, reverse=True, lambda_=4.0)
        head2.classifier.load_state_dict(head.classifier.state_dict())
        feats2 = feats.detach().clone().requires_grad_(True)
        head2(feats2).sum().backward()

        assert torch.allclose(feats2.grad, 2.0 * grad_at_2, atol=1e-5)
