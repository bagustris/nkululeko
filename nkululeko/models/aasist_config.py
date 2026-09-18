"""Centralized [AASIST] config reads for AasistModel.

Mirrors nkululeko/models/finetune_config.py's pattern: one dataclass, one
from_util() classmethod, so every [AASIST] key is resolved in one place.
MODEL.learning_rate / MODEL.optimizer / MODEL.weight_decay / MODEL.loss /
MODEL.class_weight / MODEL.patience / EXP.epochs stay in their existing
shared sections rather than [AASIST] -- they're read the same way ADM
reads them, for direct comparability between the two model types.
"""

import dataclasses


@dataclasses.dataclass
class AasistConfig:
    """Resolved [AASIST] settings, one field per config key."""

    device: str
    ssl_model: str
    max_len: int
    batch_size: int
    rawboost_algo: int
    rawboost_n_f: int
    rawboost_n_bands: int
    rawboost_min_f: int
    rawboost_max_f: int
    rawboost_min_bw: int
    rawboost_max_bw: int
    rawboost_min_coeff: int
    rawboost_max_coeff: int
    rawboost_min_g: int
    rawboost_max_g: int
    rawboost_min_bias_lin_nonlin: int
    rawboost_max_bias_lin_nonlin: int
    rawboost_p: int
    rawboost_g_sd: int
    rawboost_snr_min: int
    rawboost_snr_max: int
    domain_balanced_sampling: bool
    ssl_layer_pooling: str
    freeze_ssl_frontend: bool

    @classmethod
    def from_util(cls, util) -> "AasistConfig":
        """Build from an experiment Util, resolving all [AASIST] keys.

        `util` is a plain parameter (not `self.util`), matching
        FinetuneConfig.from_util, so this stays a pure function - testable
        without an AasistModel/experiment.

        device reads from the shared [MODEL] section (not [AASIST]) --
        matching every other shared MODEL.* key this class deliberately
        does NOT duplicate (see module docstring). An earlier version of
        this field independently resolved AASIST.device, which
        AasistModel then silently never used (it built self.device from
        MODEL.device before self.cfg existed) -- two mechanisms for the
        same setting, only one of them live. Fixed by having this one
        mechanism be the real one: AasistModel now sets self.device from
        self.cfg.device directly.
        """
        import torch

        raw_device = util.config_val("MODEL", "device", False)
        device = (
            raw_device
            if raw_device
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # ssl_model: the HuggingFace frontend checkpoint. xls-r-300m matches
        # the upstream AASIST paper's frontend (via fairseq there); swapped
        # to HuggingFace's Wav2Vec2Model here to avoid a fairseq dependency
        # (see model_aasist_core.py's HFWav2Vec2Frontend docstring).
        ssl_model = util.config_val(
            "AASIST", "ssl_model", "facebook/wav2vec2-xls-r-300m"
        )

        # max_len: fixed waveform length in samples every clip is
        # padded/truncated to. 64600 (~4.0375s at 16kHz) matches upstream's
        # own default, tuned for ASVspoof-style short utterances.
        max_len = int(util.config_val("AASIST", "max_len", "64600"))

        batch_size = int(util.config_val("AASIST", "batch_size", "24"))

        # ssl_layer_pooling: "last" (default, every earlier AASIST run
        # this session used only the final encoder layer) or "weighted"
        # (a learnable softmax-normalized combination of every hidden
        # state layer -- see HFWav2Vec2Frontend's docstring).
        ssl_layer_pooling = util.config_val("AASIST", "ssl_layer_pooling", "last")
        if ssl_layer_pooling not in ("last", "weighted"):
            util.error(
                f"unknown AASIST.ssl_layer_pooling: {ssl_layer_pooling}; "
                "expected 'last' or 'weighted'"
            )

        # freeze_ssl_frontend: skip training the SSL frontend's own
        # parameters entirely (see HFWav2Vec2Frontend's docstring for why
        # this is also a speed win, not just a regularization choice).
        freeze_ssl_frontend = util.config_val_bool(
            "AASIST", "freeze_ssl_frontend", False
        )

        # rawboost_algo: 0 disables RawBoost entirely (the "bare AASIST"
        # skeleton default); 1-8 select upstream's algorithm numbering (see
        # aasist_rawboost.apply_rawboost's docstring).
        rawboost_algo = int(util.config_val("AASIST", "rawboost_algo", "0"))

        # domain_balanced_sampling: draw each training batch with equal
        # representation from every source_db domain in the pool (see
        # aasist_sampler.DomainBalancedBatchSampler), instead of plain
        # shuffling -- default off, matching the bare-AASIST baseline.

        # RawBoost hyperparameters: defaults copied verbatim from upstream's
        # main_SSL_LA.py argparse defaults (the published ASVspoof2021
        # baseline configuration), not re-tuned here.
        return cls(
            device=device,
            ssl_model=ssl_model,
            max_len=max_len,
            batch_size=batch_size,
            ssl_layer_pooling=ssl_layer_pooling,
            freeze_ssl_frontend=freeze_ssl_frontend,
            rawboost_algo=rawboost_algo,
            rawboost_n_f=int(util.config_val("AASIST", "rawboost_n_f", "5")),
            rawboost_n_bands=int(util.config_val("AASIST", "rawboost_n_bands", "5")),
            rawboost_min_f=int(util.config_val("AASIST", "rawboost_min_f", "20")),
            rawboost_max_f=int(util.config_val("AASIST", "rawboost_max_f", "8000")),
            rawboost_min_bw=int(util.config_val("AASIST", "rawboost_min_bw", "100")),
            rawboost_max_bw=int(util.config_val("AASIST", "rawboost_max_bw", "1000")),
            rawboost_min_coeff=int(
                util.config_val("AASIST", "rawboost_min_coeff", "10")
            ),
            rawboost_max_coeff=int(
                util.config_val("AASIST", "rawboost_max_coeff", "100")
            ),
            rawboost_min_g=int(util.config_val("AASIST", "rawboost_min_g", "0")),
            rawboost_max_g=int(util.config_val("AASIST", "rawboost_max_g", "0")),
            rawboost_min_bias_lin_nonlin=int(
                util.config_val("AASIST", "rawboost_min_bias_lin_nonlin", "5")
            ),
            rawboost_max_bias_lin_nonlin=int(
                util.config_val("AASIST", "rawboost_max_bias_lin_nonlin", "20")
            ),
            rawboost_p=int(util.config_val("AASIST", "rawboost_p", "10")),
            rawboost_g_sd=int(util.config_val("AASIST", "rawboost_g_sd", "2")),
            rawboost_snr_min=int(util.config_val("AASIST", "rawboost_snr_min", "10")),
            rawboost_snr_max=int(util.config_val("AASIST", "rawboost_snr_max", "40")),
            domain_balanced_sampling=util.config_val_bool(
                "AASIST", "domain_balanced_sampling", False
            ),
        )
