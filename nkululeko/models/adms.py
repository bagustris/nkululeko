# adms.py
"""
Artifact Detection Modules (ADMs) for generalizable audio deepfake detection.

This implementation detects *artifact strength* instead of performing
binary fake/real classification. It is designed for:
- real-only or artifact-augmented training
- strong generalization to unseen generators
- clean integration into Nkululeko

Modules:
- TimeADM    : temporal / micro-prosodic artifacts
- SpectralADM: spectral / vocoder artifacts
- PhaseADM   : phase-dynamics inconsistencies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------
# Time-domain Artifact Detector
# --------------------------------------------------
class TimeADM(nn.Module):
    """
    Detects temporal artifacts using SSL features + temporal derivatives.

    Input:
        x : (B, D, T)
    Output:
        artifact score : (B, 1)
    """

    def __init__(self, feat_dim, hidden_dim=256):
        super().__init__()
        self.conv1 = nn.Conv1d(feat_dim * 2, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (B, D, T)
        if x.dim() == 2:
            x = x.unsqueeze(-1)

        # temporal derivative (micro-prosody instability)
        dx = x[:, :, 1:] - x[:, :, :-1]
        dx = F.pad(dx, (1, 0))

        x = torch.cat([x, dx], dim=1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


# --------------------------------------------------
# Spectral Artifact Detector
# --------------------------------------------------
class SpectralADM(nn.Module):
    """
    Detects spectral artifacts using spectrogram-like inputs.

    Input:
        x : (B, F, T)  (e.g., log-mel, CQCC)
    Output:
        artifact score : (B, 1)
    """

    def __init__(self, hidden_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, hidden_dim, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (B, F, T)
        assert x.dim() == 3, "SpectralADM requires (B, F, T)"

        x = x.unsqueeze(1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1).squeeze(-1)
        return self.fc(x)


# --------------------------------------------------
# Phase Artifact Detector
# --------------------------------------------------
class PhaseADM(nn.Module):
    """
    Detects phase-dynamics artifacts using BiLSTM.

    Input:
        x : (B, T, D_phase)
    Output:
        artifact score : (B, 1)
    """

    def __init__(self, feat_dim, hidden_dim=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            feat_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        # x: (B, T, D)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # phase dynamics normalization (identity suppression)
        x = x - x.mean(dim=1, keepdim=True)
        x = x / (x.std(dim=1, keepdim=True) + 1e-5)

        _, (h_n, _) = self.lstm(x)
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        return self.fc(h)


# --------------------------------------------------
# Multi-stream Artifact Detection Model
# --------------------------------------------------
class DeepfakeADMModel(nn.Module):
    """
    Multi-stream Artifact Detection Model.

    Outputs a single artifact score per utterance.
    """

    def __init__(
        self,
        ssl_feat_dim,
        phase_feat_dim,
        fusion="weighted",
    ):
        super().__init__()
        self.fusion = fusion

        self.time_adm = TimeADM(ssl_feat_dim)
        self.spec_adm = SpectralADM()
        self.phase_adm = PhaseADM(phase_feat_dim)

        if fusion == "weighted":
            self.weights = nn.Parameter(torch.ones(3))
        elif fusion == "concat":
            self.fusion_fc = nn.Linear(3, 1)

    def forward(self, ssl_feats, spec_feats, phase_feats):
        """
        Args:
            ssl_feats   : (B, D, T)
            spec_feats  : (B, F, T)
            phase_feats : (B, T, Dp)

        Returns:
            artifact_score : (B,)
        """
        t = self.time_adm(ssl_feats)
        s = self.spec_adm(spec_feats)
        p = self.phase_adm(phase_feats)

        scores = torch.cat([t, s, p], dim=1)

        if self.fusion == "avg":
            out = scores.mean(dim=1, keepdim=True)
        elif self.fusion == "max":
            out = scores.max(dim=1, keepdim=True)[0]
        elif self.fusion == "weighted":
            w = F.softmax(self.weights, dim=0)
            out = w[0] * t + w[1] * s + w[2] * p
        elif self.fusion == "concat":
            out = self.fusion_fc(scores)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion}")

        return out.squeeze(1)

    @torch.no_grad()
    def score(self, ssl_feats, spec_feats, phase_feats):
        return self.forward(ssl_feats, spec_feats, phase_feats)
