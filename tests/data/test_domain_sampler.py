"""Unit tests for DomainBalancedBatchSampler (nkululeko/models/aasist_sampler.py)."""

import numpy as np
import pandas as pd
import pytest

from nkululeko.models.aasist_sampler import DomainBalancedBatchSampler


def _df(counts):
    """Build a df with a source_db column: counts is {domain: n_rows}."""
    rows = []
    for domain, n in counts.items():
        rows.extend([domain] * n)
    return pd.DataFrame({"source_db": rows})


class TestValidation:
    def test_missing_source_db_column_raises(self):
        df = pd.DataFrame({"label": [0, 1]})
        with pytest.raises(ValueError, match="source_db"):
            DomainBalancedBatchSampler(df, batch_size=4)

    def test_single_domain_raises(self):
        df = _df({"a": 10})
        with pytest.raises(ValueError, match=">=2 domains"):
            DomainBalancedBatchSampler(df, batch_size=4)


class TestEqualSizedDomains:
    def test_every_batch_has_equal_representation(self):
        df = _df({"a": 20, "b": 20, "c": 20, "d": 20})
        sampler = DomainBalancedBatchSampler(df, batch_size=16, seed=0)
        domain_of = df["source_db"].to_numpy()

        for batch in sampler:
            assert len(batch) == 16
            counts = pd.Series(domain_of[batch]).value_counts()
            assert set(counts.index) == {"a", "b", "c", "d"}
            assert (counts == 4).all()

    def test_len_matches_iteration_count(self):
        df = _df({"a": 20, "b": 20, "c": 20, "d": 20})
        sampler = DomainBalancedBatchSampler(df, batch_size=16, seed=0)
        assert len(sampler) == len(list(sampler))

    def test_every_row_appears_within_one_epoch(self):
        df = _df({"a": 8, "b": 8})
        sampler = DomainBalancedBatchSampler(df, batch_size=4, seed=0)
        seen = sorted(i for batch in sampler for i in batch)
        assert seen == list(range(16))


class TestUnequalSizedDomains:
    def test_smaller_domain_is_cycled_to_match_larger(self):
        # domain "a" has 40 rows, "b" has only 10 -- "b" must be cycled
        # (reshuffled and repeated) to cover the same epoch length as "a".
        df = _df({"a": 40, "b": 10})
        sampler = DomainBalancedBatchSampler(df, batch_size=4, seed=0)
        domain_of = df["source_db"].to_numpy()

        b_appearances = 0
        for batch in sampler:
            counts = pd.Series(domain_of[batch]).value_counts()
            assert counts.get("a", 0) == 2
            assert counts.get("b", 0) == 2
            b_appearances += counts.get("b", 0)

        # "b" only has 10 unique rows but must appear len(sampler)*2 times
        # (matching "a"'s epoch length) -- more than 10, proving it cycled.
        assert b_appearances > 10

    def test_every_b_row_appears_at_least_once_per_epoch(self):
        """A cycled (repeat+reshuffle) domain should cover all its own
        rows at least once before any row repeats a second time -- unlike
        uniform-random-with-replacement sampling, which could skip rows."""
        df = _df({"a": 40, "b": 10})
        sampler = DomainBalancedBatchSampler(df, batch_size=4, seed=1)
        domain_of = df["source_db"].to_numpy()
        b_offset = 40  # "b" rows are positions 40..49

        seen_b = set()
        for batch in sampler:
            for i in batch:
                if domain_of[i] == "b":
                    seen_b.add(i)

        assert seen_b == set(range(b_offset, b_offset + 10))


class TestReproducibility:
    def test_same_seed_gives_same_batches(self):
        df = _df({"a": 20, "b": 20})
        s1 = DomainBalancedBatchSampler(df, batch_size=8, seed=42)
        s2 = DomainBalancedBatchSampler(df, batch_size=8, seed=42)
        assert list(s1) == list(s2)

    def test_no_seed_gives_different_batches_across_iterations(self):
        df = _df({"a": 50, "b": 50})
        sampler = DomainBalancedBatchSampler(df, batch_size=8, seed=None)
        first = list(sampler)
        second = list(sampler)
        assert first != second
