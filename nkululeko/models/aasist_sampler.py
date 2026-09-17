"""Domain-balanced batch sampling for AasistModel's training loader.

Motivation: the full LODO training pool (not the stratified-equal
subsample this session's AASIST baselines actually train on) has wildly
uneven per-database row counts -- LA19 alone is ~121k rows against itw's
~32k in the full pool -- so plain shuffling lets the largest pooled
dataset dominate batch gradients by sheer count. DomainBalancedBatchSampler
draws (as close to) equal representation from every source_db domain in
every batch, cycling (reshuffling and repeating) smaller domains to match
the largest domain's per-epoch length -- the standard round-robin
balanced-batch strategy from domain-adaptation training.

Requires the source_db column Datasplitter.fill_train_and_tests() now
stamps onto every pooled row (see nkululeko/data/datasplitter.py) --
without it there is no domain to balance by.
"""

import numpy as np
from torch.utils.data import Sampler


class DomainBalancedBatchSampler(Sampler):
    """Yields batches (lists of positional indices into `df`) with equal
    representation from each unique value of df["source_db"].

    Use via DataLoader(dataset, batch_sampler=DomainBalancedBatchSampler(...))
    -- batch_sampler is mutually exclusive with DataLoader's own
    batch_size/shuffle/sampler arguments.
    """

    def __init__(self, df, batch_size, seed=None):
        if "source_db" not in df.columns:
            raise ValueError(
                "DomainBalancedBatchSampler requires a source_db column "
                f"(added by Datasplitter when pooling multiple databases); "
                f"got columns: {list(df.columns)}"
            )
        domains = sorted(df["source_db"].dropna().unique().tolist())
        if len(domains) < 2:
            raise ValueError(
                f"DomainBalancedBatchSampler needs >=2 domains to balance "
                f"across, got: {domains}"
            )
        self.domains = domains
        self.domain_indices = {
            d: np.flatnonzero((df["source_db"] == d).to_numpy()) for d in domains
        }
        # per_domain: rows drawn from each domain per batch. Rounds
        # batch_size down to a multiple of len(domains) if it doesn't
        # divide evenly (e.g. batch_size=16, 4 domains -> 4/domain, exact;
        # batch_size=17, 4 domains -> 4/domain, 16 actual batch size).
        self.per_domain = max(1, batch_size // len(domains))
        self.epoch_len = max(len(idx) for idx in self.domain_indices.values())
        self.n_batches = -(-self.epoch_len // self.per_domain)  # ceil div
        self._seed = seed

    def __len__(self):
        return self.n_batches

    def _cycled_shuffled(self, indices, length, rng):
        """Repeat+reshuffle `indices` (each full pass independently
        shuffled) until at least `length` long, then truncate -- so a
        smaller domain's rows each appear an equal number of times (plus
        at most one extra) rather than being sampled with replacement
        uniformly at random, which would let a few rows repeat far more
        than others within one epoch.
        """
        reps = -(-length // len(indices))  # ceil div
        pieces = [rng.permutation(indices) for _ in range(reps)]
        return np.concatenate(pieces)[:length]

    def __iter__(self):
        rng = np.random.default_rng(self._seed)
        needed = self.n_batches * self.per_domain
        per_domain_order = {
            d: self._cycled_shuffled(idx, needed, rng)
            for d, idx in self.domain_indices.items()
        }
        for b in range(self.n_batches):
            start = b * self.per_domain
            end = start + self.per_domain
            batch = []
            for d in self.domains:
                batch.extend(per_domain_order[d][start:end].tolist())
            rng.shuffle(batch)
            yield batch
