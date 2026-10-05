"""Tests for deliberate fixes to the ported AASIST backend."""

import torch

from nkululeko.models.model_aasist_core import ResidualBlock


def test_non_first_residual_block_applies_preactivation():
    torch.manual_seed(0)
    block = ResidualBlock([4, 4]).eval()
    x = torch.randn(2, 4, 8, 8) * 5 + 3
    out = block(x)
    expected = block.conv2(block.selu(block.bn2(block.conv1(block.selu(block.bn1(x))))))
    assert torch.allclose(out, expected + x, atol=1e-5)


def test_first_residual_block_skips_preactivation():
    torch.manual_seed(0)
    block = ResidualBlock([4, 4], first=True).eval()
    x = torch.randn(2, 4, 8, 8)
    expected = block.conv2(block.selu(block.bn2(block.conv1(x))))
    assert torch.allclose(block(x), expected + x, atol=1e-5)
