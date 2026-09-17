"""Unit tests for the ported AASIST backend (nkululeko/models/model_aasist_core.py).

Builds a tiny randomly-initialized Wav2Vec2Config-based model directly
(no network/download -- same technique as test_model_tuned_backbone.py)
and monkeypatches Wav2Vec2Model.from_pretrained to return it, so
AasistBackend's graph-attention pipeline is exercised end-to-end on real
(if tiny) SSL frontend output shapes without needing the real 300M-param
XLS-R checkpoint.
"""

import torch
from transformers import Wav2Vec2Config, Wav2Vec2Model

from nkululeko.models import model_aasist_core as core


def _tiny_wav2vec2(hidden_size=32):
    config = Wav2Vec2Config(
        hidden_size=hidden_size,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        conv_dim=(16, 16, 16, 16, 16, 16, 16),
        conv_stride=(5, 2, 2, 2, 2, 2, 2),
        conv_kernel=(10, 3, 3, 3, 3, 2, 2),
        attn_implementation="eager",
    )
    return Wav2Vec2Model(config)


class TestHFWav2Vec2Frontend:
    def test_out_dim_matches_checkpoint_hidden_size(self, monkeypatch):
        tiny = _tiny_wav2vec2(hidden_size=32)
        monkeypatch.setattr(Wav2Vec2Model, "from_pretrained", lambda *a, **k: tiny)

        frontend = core.HFWav2Vec2Frontend("dummy/checkpoint")

        assert frontend.out_dim == 32

    def test_extract_feat_squeezes_trailing_channel_dim(self, monkeypatch):
        tiny = _tiny_wav2vec2(hidden_size=32)
        monkeypatch.setattr(Wav2Vec2Model, "from_pretrained", lambda *a, **k: tiny)
        frontend = core.HFWav2Vec2Frontend("dummy/checkpoint")
        frontend.eval()  # disable dropout so 2D/3D inputs are directly comparable

        x_2d = torch.randn(2, 16000)
        x_3d = x_2d.unsqueeze(-1)

        with torch.no_grad():
            feat_2d = frontend.extract_feat(x_2d)
            feat_3d = frontend.extract_feat(x_3d)

        assert feat_2d.ndim == 3  # (batch, frames, hidden)
        assert feat_2d.shape[0] == 2
        assert feat_2d.shape[-1] == 32
        assert torch.equal(feat_2d, feat_3d)


class TestAasistBackendForward:
    def test_forward_returns_two_class_logits(self, monkeypatch):
        tiny = _tiny_wav2vec2(hidden_size=32)
        monkeypatch.setattr(Wav2Vec2Model, "from_pretrained", lambda *a, **k: tiny)

        backend = core.AasistBackend("dummy/checkpoint")
        backend.eval()

        x = torch.randn(3, 16000)  # 1 second at 16kHz, well under max_len
        with torch.no_grad():
            logits = backend(x)

        assert logits.shape == (3, 2)
        assert torch.all(torch.isfinite(logits))

    def test_forward_handles_trailing_channel_dim_input(self, monkeypatch):
        tiny = _tiny_wav2vec2(hidden_size=32)
        monkeypatch.setattr(Wav2Vec2Model, "from_pretrained", lambda *a, **k: tiny)

        backend = core.AasistBackend("dummy/checkpoint")
        backend.eval()

        x = torch.randn(2, 16000, 1)
        with torch.no_grad():
            logits = backend(x)

        assert logits.shape == (2, 2)


class TestGraphSubmodules:
    def test_graph_attention_layer_preserves_node_count(self):
        layer = core.GraphAttentionLayer(in_dim=8, out_dim=4)
        x = torch.randn(2, 5, 8)
        out = layer(x)
        assert out.shape == (2, 5, 4)

    def test_graph_pool_keeps_ratio_of_nodes(self):
        pool = core.GraphPool(k=0.5, in_dim=4, p=0.0)
        h = torch.randn(2, 10, 4)
        out = pool(h)
        assert out.shape == (2, 5, 4)

    def test_residual_block_changes_channels_via_downsample(self):
        block = core.ResidualBlock(nb_filts=[1, 32], first=True)
        x = torch.randn(2, 1, 20, 20)
        out = block(x)
        assert out.shape[1] == 32

    def test_htrg_graph_attention_layer_shapes(self):
        layer = core.HtrgGraphAttentionLayer(in_dim=8, out_dim=4)
        x1 = torch.randn(2, 3, 8)
        x2 = torch.randn(2, 3, 8)
        out1, out2, master = layer(x1, x2)
        assert out1.shape == (2, 3, 4)
        assert out2.shape == (2, 3, 4)
        assert master.shape == (2, 1, 4)
