"""AasistModel: [MODEL] type = aasist.

Waveform-level end-to-end AASIST (SSL frontend + spectro-temporal graph
attention backend, see model_aasist_core.py) -- a genuinely different
pipeline shape from every other model in this project: it reads raw
audio directly from df_train/df_test (via audiofile.read(), same pattern
as model_tuned.py) and bypasses nkululeko's FeatureExtractor entirely, so
a config using this model must set FEATS.type = [] (no features to
extract). model_type is set to "ann" (not "finetuned"): unlike
TunedModel's HuggingFace Trainer, AASIST's train() runs exactly one
epoch per call, so modelrunner.do_epochs()'s own per-epoch loop (and
therefore MODEL.patience-based early stopping, EXP.traindevtest's
dev/test split, and every other existing epoch-level machinery) applies
identically to how it does for model_adm.py -- the model this is meant
to be directly comparable against.

RawBoost augmentation (MODEL section: AASIST.rawboost_algo, 0 = off) is
applied to the *train* split only, before pad/truncate, inside
_WaveformDataset.__getitem__ -- see aasist_rawboost.py.

Domain-balanced batch sampling (MODEL.domain_balanced_sampling, off by
default, shared with ADMModel) replaces plain shuffling on the train
loader with DomainBalancedBatchSampler (nkululeko/data/domain_sampler.py,
model-agnostic), which draws equal representation from every source_db
domain in every batch -- see that module's docstring for why.

Domain-adversarial training (MODEL.dann_columns, empty/off by default)
attaches one nkululeko.models.domain_adversarial.DomainAdversarialHead
per listed nuisance column (e.g. source_db) to the backend's pooled
feature vector (AasistBackend.forward(..., return_features=True)) --
only on the *train* split, mirroring RawBoost/domain-balanced sampling's
own augment-only gating in get_loader(). _WaveformDataset returns a
3-tuple (waveform, label, domain_labels) instead of the default 2-tuple
only when both augment=True and cfg.dann_columns is non-empty, so
dev/test loaders and the no-DANN default path are byte-identical to
before.
"""

import itertools

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import recall_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset

from nkululeko.data.domain_sampler import DomainBalancedBatchSampler
from nkululeko.models.aasist_config import AasistConfig
from nkululeko.models.aasist_rawboost import apply_rawboost
from nkululeko.models.domain_adversarial import DomainAdversarialHead
from nkululeko.models.model import Model
from nkululeko.models.model_aasist_core import AasistBackend
from nkululeko.optimizers import (
    get_optimizer,
    get_scheduler,
    initialize_cosine_scheduler,
    step_scheduler,
)
from nkululeko.optimizers.sam import is_sam_optimizer
from nkululeko.reporting.reporter import Reporter


def _pad_or_tile(signal, max_len):
    """Pad-by-tiling (upstream's convention) or truncate to a fixed length."""
    sig_len = signal.shape[0]
    if sig_len >= max_len:
        return signal[:max_len]
    num_repeats = int(max_len / sig_len) + 1
    return np.tile(signal, num_repeats)[:max_len]


class _WaveformDataset(Dataset):
    """Reads raw waveforms directly from disk, bypassing FeatureExtractor.

    Mirrors model_tuned.py's own raw-audio loading (audiofile.read() off
    the (file, start, end) segmented index), including its NaT handling
    for whole-file (non-segmented) rows -- except using `pd.isna(end)`
    rather than `end == pd.NaT` (which is always False; pd.NaT compares
    unequal to everything, including itself -- the same bug Felix's
    "bugfix: non-unique feature cache names" commit already fixed
    elsewhere in this codebase, in feats_audwav2vec2.py/feats_audmodel.py).
    """

    def __init__(self, df, target, cfg, augment, dann_label_maps=None):
        self.df = df
        self.target = target
        self.cfg = cfg
        self.augment = augment
        # dann_label_maps: {column: {raw_value: int_index}}, built once by
        # AasistModel.__init__ from df_train -- only set (non-None) when
        # this dataset should emit the 3-tuple form (train split, DANN on).
        self.dann_label_maps = dann_label_maps if (augment and cfg.dann_columns) else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        import audiofile

        file, start, end = self.df.index[idx]
        if pd.isna(end):
            signal, sr = audiofile.read(file, offset=start)
        else:
            signal, sr = audiofile.read(file, duration=end - start, offset=start)
        signal = np.asarray(signal).squeeze()

        if self.augment and self.cfg.rawboost_algo:
            signal = apply_rawboost(signal, sr, self.cfg, self.cfg.rawboost_algo)

        signal = _pad_or_tile(signal, self.cfg.max_len)
        row = self.df.iloc[idx]
        label = row[self.target]
        waveform = torch.tensor(signal, dtype=torch.float32)
        if self.dann_label_maps is None:
            return waveform, label

        domain_labels = torch.tensor(
            [self.dann_label_maps[col][row[col]] for col in self.cfg.dann_columns],
            dtype=torch.long,
        )
        return waveform, label, domain_labels


class AasistModel(Model):
    """AASIST = spectro-temporal graph attention network for deepfake detection."""

    is_classifier = True

    def __init__(self, df_train, df_test, feats_train, feats_test, context=None):
        super().__init__(df_train, df_test, feats_train, feats_test, context=context)
        super().set_model_type("ann")
        self.name = "aasist"
        self.target = self.context.config["DATA"]["target"]

        manual_seed = eval(self.util.config_val("MODEL", "random_seed", "False"))
        if manual_seed:
            self.util.debug(f"seeding random to {manual_seed}")
            torch.manual_seed(int(manual_seed))

        labels = self.context.labels
        self.class_num = len(labels)
        if self.class_num != 2:
            self.util.error(
                f"aasist model requires exactly 2 classes (got {self.class_num}: "
                f"{labels}) -- its output layer is a fixed binary (real/fake) head"
            )

        self.cfg = AasistConfig.from_util(self.util)
        self.device = self.cfg.device
        self.util.debug(
            f"aasist: SSL frontend {self.cfg.ssl_model}, max_len={self.cfg.max_len}, "
            f"rawboost_algo={self.cfg.rawboost_algo}"
        )
        self.net = AasistBackend(
            self.cfg.ssl_model,
            layer_pooling=self.cfg.ssl_layer_pooling,
            freeze_ssl=self.cfg.freeze_ssl_frontend,
        ).to(self.device)

        self._build_criterion(df_train)
        self._build_dann_heads(df_train)

        dann_params = (
            itertools.chain(self.net.parameters(), self.dann_heads.parameters())
            if self.cfg.dann_columns
            else self.net.parameters()
        )
        self.optimizer, self.learning_rate = get_optimizer(
            dann_params, self.util, default_lr=1e-5, default_optimizer="adam"
        )
        self.scheduler, self.scheduler_type, self.scheduler_needs_init = get_scheduler(
            self.optimizer, self.util, default_scheduler="none"
        )

        self.trainloader = self.get_loader(df_train, augment=True, shuffle=True)
        self.testloader = self.get_loader(df_test, augment=False, shuffle=False)

    def _build_dann_heads(self, df_train):
        """Build one DomainAdversarialHead per MODEL.dann_columns entry,
        attached to the backend's pooled feature vector (self.net.feat_dim
        wide). Label maps are built once from df_train's own values (not
        the global label set) -- fine since DANN heads only ever run on
        the train split (see _WaveformDataset's augment-gated 3-tuple)."""
        self.dann_label_maps = {}
        if not self.cfg.dann_columns:
            self.dann_heads = None
            return
        heads = {}
        for col in self.cfg.dann_columns:
            values = sorted(df_train[col].dropna().unique().tolist())
            if len(values) < 2:
                self.util.error(
                    f"MODEL.dann_columns includes '{col}', but df_train has "
                    f"{len(values)} unique value(s) for it -- DANN needs >=2 "
                    "classes to discriminate against."
                )
            self.dann_label_maps[col] = {v: i for i, v in enumerate(values)}
            heads[col] = DomainAdversarialHead(
                feat_dim=self.net.feat_dim,
                num_classes=len(values),
                reverse=self.cfg.dann_reverse,
                lambda_=self.cfg.dann_lambda,
            )
        self.dann_heads = nn.ModuleDict(heads).to(self.device)
        self.util.debug(
            f"aasist: DANN heads for {self.cfg.dann_columns} "
            f"(reverse={self.cfg.dann_reverse}, lambda={self.cfg.dann_lambda}, "
            f"weight={self.cfg.dann_weight})"
        )

    def _build_criterion(self, df_train):
        """CrossEntropyLoss over the fixed 2-way output, with optional
        MODEL.class_weight="auto" balancing (matching ADM's own
        MODEL.class_weight convention, adapted for a 2-logit softmax head
        instead of ADM's single-logit sigmoid). Only MODEL.loss=cross is
        supported for this first (no-generalization-technique) skeleton.
        """
        loss_type = self.util.config_val("MODEL", "loss", "cross")
        if loss_type != "cross":
            self.util.error(
                f"aasist model currently only supports MODEL.loss=cross (got '{loss_type}')"
            )
        label_smoothing = self._get_label_smoothing()
        weight = None
        if self.util.config_val("MODEL", "class_weight", False):
            classes = np.arange(self.class_num)
            cw = compute_class_weight(
                class_weight="balanced", classes=classes, y=df_train[self.target]
            )
            weight = torch.tensor(cw, dtype=torch.float32, device=self.device)
            self.util.debug(f"aasist: class weights {cw}")
        self.criterion = torch.nn.CrossEntropyLoss(
            weight=weight, label_smoothing=label_smoothing
        )

    def get_loader(self, df, augment, shuffle):
        dataset = _WaveformDataset(
            df,
            self.target,
            self.cfg,
            augment=augment,
            dann_label_maps=self.dann_label_maps,
        )
        # Each __getitem__ does its own audiofile.read() (+ optional
        # RawBoost, which is pure-numpy/scipy FIR filtering) -- CPU-bound
        # work that a single-process loader (num_workers=0) serializes
        # with GPU compute. MODEL.n_jobs (already read by the base Model
        # class into self.n_jobs) parallelizes it the same way ADM's
        # get_loader could but doesn't need to (ADM's TensorDataset has no
        # per-item work). persistent_workers avoids respawning the worker
        # pool every epoch when num_workers > 0.
        loader_kwargs = {}
        if self.n_jobs > 0:
            loader_kwargs["num_workers"] = self.n_jobs
            loader_kwargs["persistent_workers"] = True
        # augment=True uniquely marks the training split (see __init__ and
        # reset_test/set_testdata below) -- domain-balanced sampling only
        # ever makes sense for training batches; dev/test come from a
        # single held-out domain in this project's fold designs anyway.
        if augment and self.cfg.domain_balanced_sampling:
            sampler = DomainBalancedBatchSampler(df, self.cfg.batch_size)
            return DataLoader(dataset, batch_sampler=sampler, **loader_kwargs)
        return DataLoader(
            dataset, batch_size=self.cfg.batch_size, shuffle=shuffle, **loader_kwargs
        )

    def set_testdata(self, data_df, feats_df):
        self.df_test, self.feats_test = data_df, feats_df
        self.testloader = self.get_loader(data_df, augment=False, shuffle=False)

    def reset_test(self, df_test, feats_test):
        self.df_test, self.feats_test = df_test, feats_test
        self.testloader = self.get_loader(df_test, augment=False, shuffle=False)

    def train(self):
        """Train for exactly one epoch (do_epochs() calls this once per
        epoch and owns the epoch loop/patience, matching model_adm.py)."""
        if self.scheduler_needs_init and self.scheduler is None:
            self.scheduler = initialize_cosine_scheduler(
                self.optimizer, self.util, steps_per_epoch=len(self.trainloader)
            )
            self.scheduler_needs_init = False

        self.net.train()
        losses = []
        sam_active = is_sam_optimizer(self.optimizer)
        dann_active = bool(self.cfg.dann_columns)
        for batch in self.trainloader:
            if dann_active:
                waveforms, labels, domain_labels = batch
                domain_labels = domain_labels.to(self.device)
            else:
                waveforms, labels = batch
                domain_labels = None
            waveforms = waveforms.to(self.device)
            labels = labels.long().to(self.device)

            if sam_active:
                # SAM needs two forward/backward passes per step (see
                # nkululeko.optimizers.sam's docstring) -- the closure is
                # called once at the current weights (ascent direction)
                # and once at the perturbed point (the gradient the base
                # optimizer actually updates with).
                def closure():
                    self.optimizer.zero_grad()
                    loss = self._aasist_forward_loss(waveforms, labels, domain_labels)
                    loss.backward()
                    return loss

                loss = self.optimizer.step(closure)
            else:
                loss = self._aasist_forward_loss(waveforms, labels, domain_labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            step_scheduler(self.scheduler, self.scheduler_type, step_per_batch=True)
            losses.append(loss.item())

        step_scheduler(self.scheduler, self.scheduler_type, step_per_batch=False)
        self.loss = float(np.mean(losses)) if losses else 0.0

    def _aasist_forward_loss(self, waveforms, labels, domain_labels):
        """One forward pass + loss computation -- factored out so the SAM
        closure above can call it twice per step (see train()), and so
        DANN's extra per-column adversarial loss terms don't need a
        separate SAM/plain implementation. `domain_labels` is None
        whenever DANN is off (see train()'s dann_active branch)."""
        if self.dann_heads is None:
            return self.criterion(self.net(waveforms), labels)

        logits, feats = self.net(waveforms, return_features=True)
        loss = self.criterion(logits, labels)
        for i, col in enumerate(self.cfg.dann_columns):
            dann_logits = self.dann_heads[col](feats)
            loss = loss + self.cfg.dann_weight * torch.nn.functional.cross_entropy(
                dann_logits, domain_labels[:, i]
            )
        return loss

    def evaluate(self, loader):
        self.net.eval()
        all_logits, all_targets, losses = [], [], []
        with torch.no_grad():
            for waveforms, labels in loader:
                waveforms = waveforms.to(self.device)
                labels_t = labels.long().to(self.device)
                logits = self.net(waveforms)
                losses.append(self.criterion(logits, labels_t).item())
                all_logits.append(logits.cpu())
                all_targets.append(labels.cpu())

        logits = torch.cat(all_logits) if all_logits else torch.empty(0, 2)
        targets = torch.cat(all_targets) if all_targets else torch.empty(0)
        predictions = torch.argmax(logits, dim=1)
        uar = (
            recall_score(targets.numpy(), predictions.numpy(), average="macro")
            if len(targets)
            else 0.0
        )
        return (
            uar,
            targets,
            predictions,
            logits,
            float(np.mean(losses)) if losses else 0.0,
        )

    def get_probas(self, logits):
        probs = torch.softmax(logits, dim=1).numpy()
        proba_d = {c: probs[:, i] for i, c in enumerate(np.arange(self.class_num))}
        probas = pd.DataFrame(proba_d)
        return probas.set_index(self.df_test.index)

    def predict(self):
        """Predict on the current test set.

        Unlike model_adm.py's predict(), this does NOT also evaluate over
        the full trainloader to report a "train" UAR figure: for a
        300M-parameter SSL-frontend model, a second full forward pass over
        every training utterance every single epoch would roughly double
        AASIST's already much higher per-epoch cost compared to ADM's
        cached-feature training, for a purely diagnostic number.
        """
        _, truths, predictions, logits, loss_eval = self.evaluate(self.testloader)
        self.loss_eval = loss_eval
        probas = self.get_probas(logits)
        report = Reporter(
            truths.numpy().astype(float),
            predictions.numpy(),
            self.run,
            self.epoch,
            probas=probas,
            context=self.context,
        )
        if hasattr(self, "loss"):
            report.result.loss = self.loss
        report.result.loss_eval = self.loss_eval
        return report

    def get_predictions(self):
        _, _, predictions, logits, _ = self.evaluate(self.testloader)
        return predictions.numpy(), self.get_probas(logits)

    def store(self):
        torch.save(self.net.state_dict(), self.store_path)

    def load(self, run, epoch):
        self.set_id(run, epoch)
        try:
            self.net.load_state_dict(
                torch.load(self.store_path, map_location=self.device, weights_only=True)
            )
        except FileNotFoundError:
            self.util.error(f"model file not found: {self.store_path}")
        self.net.eval()
