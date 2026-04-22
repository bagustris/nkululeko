"""Tests for nkululeko/data/datasplitter.py — Datasplitter class."""

import configparser
import pickle
from datetime import timedelta

import pandas as pd
import pytest

import nkululeko.glob_conf as glob_conf
from nkululeko.data.datasplitter import Datasplitter


def _make_segmented_index(files):
    arrays = [
        files,
        [timedelta(0)] * len(files),
        [timedelta(seconds=1)] * len(files),
    ]
    return pd.MultiIndex.from_arrays(arrays, names=["file", "start", "end"])


@pytest.fixture(autouse=True)
def setup_glob_conf(tmp_path):
    config = configparser.ConfigParser()
    config["EXP"] = {
        "type": "classification",
        "name": "test_ds",
        "root": str(tmp_path),
        "runs": "1",
        "epochs": "1",
        "traindevtest": "False",
    }
    config["DATA"] = {
        "target": "emotion",
        "databases": "['test_db']",
        "labels": "['happy', 'sad', 'angry']",
    }
    config["MODEL"] = {"type": "xgb"}
    config["FEATS"] = {"type": "['os']"}
    glob_conf.init_config(config)
    glob_conf.labels = ["happy", "sad", "angry"]
    glob_conf.target = "emotion"
    glob_conf.split3 = False
    yield
    glob_conf.config = None
    glob_conf.labels = None
    glob_conf.target = None


@pytest.fixture
def ds_bare(tmp_path):
    """A Datasplitter created with __new__ (no datasets needed)."""
    ds = Datasplitter.__new__(Datasplitter)
    ds.util = type(
        "U",
        (),
        {
            "get_path": lambda self, k: str(tmp_path) + "/",
            "warn": lambda self, m: None,
        },
    )()
    ds.datasets = {}
    ds.target = "emotion"
    ds.split3 = False
    ds.got_speaker = False
    return ds


class TestAddRandomTarget:
    def test_adds_target_column(self, ds_bare):
        df = pd.DataFrame({"speaker": ["s1", "s2", "s3"]})
        result = ds_bare._add_random_target(df)
        assert "emotion" in result.columns

    def test_all_labels_from_glob_labels(self, ds_bare):
        df = pd.DataFrame(index=range(100))
        result = ds_bare._add_random_target(df)
        assert set(result["emotion"]).issubset({"happy", "sad", "angry"})

    def test_returns_same_length(self, ds_bare):
        df = pd.DataFrame(index=range(10))
        result = ds_bare._add_random_target(df)
        assert len(result) == 10


class TestGetSampleSelection:
    def _make_ds_with_splits(self, tmp_path):
        from nkululeko.utils.util import Util

        ds = Datasplitter.__new__(Datasplitter)
        ds.util = Util("datasplitter")
        ds.target = "emotion"
        ds.split3 = False
        ds.got_speaker = False
        ds.datasets = {}
        idx_tr = _make_segmented_index(["/data/tr_1.wav", "/data/tr_2.wav"])
        idx_te = _make_segmented_index(["/data/te_1.wav"])
        ds.df_train = pd.DataFrame({"emotion": [0, 1]}, index=idx_tr)
        ds.df_test = pd.DataFrame({"emotion": [0]}, index=idx_te)
        return ds

    def test_all_returns_train_and_test(self, tmp_path):
        glob_conf.config["EXP"]["sample_selection"] = "all"
        ds = self._make_ds_with_splits(tmp_path)
        result = ds.get_sample_selection()
        assert len(result) == 3

    def test_train_returns_only_train(self, tmp_path):
        glob_conf.config["EXP"]["sample_selection"] = "train"
        ds = self._make_ds_with_splits(tmp_path)
        result = ds.get_sample_selection()
        assert len(result) == 2

    def test_test_returns_only_test(self, tmp_path):
        glob_conf.config["EXP"]["sample_selection"] = "test"
        ds = self._make_ds_with_splits(tmp_path)
        result = ds.get_sample_selection()
        assert len(result) == 1


class TestBuildTestDsDf:
    def test_empty_test_produces_empty_mapping(self, ds_bare):
        ds_bare.df_test = pd.DataFrame()
        ds_bare._build_test_ds_df()
        assert ds_bare.test_ds_df == {}

    def test_in_memory_split_used_preferentially(self, tmp_path):
        ds = Datasplitter.__new__(Datasplitter)
        ds.util = type("U", (), {"get_path": lambda self, k: str(tmp_path) + "/"})()
        ds.target = "emotion"
        ds.split3 = False
        ds.got_speaker = False

        files_a = ["/data/a_1.wav", "/data/a_2.wav"]
        idx_a = _make_segmented_index(files_a)
        df_a = pd.DataFrame({"emotion": [0, 1]}, index=idx_a)

        mock_ds = type("DS", (), {"df_test": df_a})()
        ds.datasets = {"db_a": mock_ds}
        ds.df_test = df_a
        ds._build_test_ds_df()

        assert "db_a" in ds.test_ds_df
        assert len(ds.test_ds_df["db_a"]) == 2

    def test_falls_back_to_pkl_when_no_in_memory(self, tmp_path):
        files = ["/data/f_1.wav", "/data/f_2.wav"]
        idx = _make_segmented_index(files)
        df = pd.DataFrame({"emotion": [0, 1]}, index=idx)
        df.to_pickle(str(tmp_path) + "/mydb_testdf.pkl")

        ds = Datasplitter.__new__(Datasplitter)
        ds.util = type("U", (), {"get_path": lambda self, k: str(tmp_path) + "/"})()
        ds.target = "emotion"
        ds.split3 = False
        ds.got_speaker = False
        ds.datasets = {"mydb": None}  # None → no in-memory split
        ds.df_test = df
        ds._build_test_ds_df()

        assert "mydb" in ds.test_ds_df

    def test_two_datasets_correctly_separated(self, tmp_path):
        files_a = ["/data/a.wav"]
        files_b = ["/data/b.wav"]
        idx_a = _make_segmented_index(files_a)
        idx_b = _make_segmented_index(files_b)
        df_a = pd.DataFrame({"emotion": [0]}, index=idx_a)
        df_b = pd.DataFrame({"emotion": [1]}, index=idx_b)
        df_test = pd.concat([df_a, df_b])

        ds_a = type("DS", (), {"df_test": df_a})()
        ds_b = type("DS", (), {"df_test": df_b})()

        ds = Datasplitter.__new__(Datasplitter)
        ds.util = type("U", (), {"get_path": lambda self, k: str(tmp_path) + "/"})()
        ds.target = "emotion"
        ds.split3 = False
        ds.got_speaker = False
        ds.datasets = {"db_a": ds_a, "db_b": ds_b}
        ds.df_test = df_test
        ds._build_test_ds_df()

        assert set(ds.test_ds_df.keys()) == {"db_a", "db_b"}
        assert len(ds.test_ds_df["db_a"]) == 1
        assert len(ds.test_ds_df["db_b"]) == 1


class TestFillTrainAndTestsEarlyReturn:
    def test_unsupervised_returns_splits_without_labels(self, tmp_path):
        """fill_train_and_tests returns (df_train, df_test) even when target is None."""

        class FakeDataset:
            def split(self):
                self.df_train = pd.DataFrame(index=range(2))
                self.df_test = pd.DataFrame(index=range(1))
                self.df_train.is_labeled = False
                self.df_test.is_labeled = False
                self.df_train.got_gender = False
                self.df_test.got_gender = False
                self.df_train.got_speaker = False
                self.df_test.got_speaker = False

            def prepare_labels(self):
                pass  # no-op: unsupervised run has no labels to encode

            df_train = pd.DataFrame()
            df_test = pd.DataFrame()
            name = "fake"

        glob_conf.config["DATA"]["target"] = "none"
        glob_conf.target = None

        fake_ds = FakeDataset()
        ds = Datasplitter.__new__(Datasplitter)
        ds.util = type(
            "U",
            (),
            {
                "get_path": lambda self, k: str(tmp_path) + "/",
                "config_val": lambda self, sec, key, default: default,
                "debug": lambda self, m: None,
                "warn": lambda self, m: None,
                "copy_flags": lambda self, src, tgt: None,
                "exp_is_classification": lambda self: False,
            },
        )()
        ds.target = None
        ds.split3 = False
        ds.got_speaker = False
        ds.datasets = {"fake": fake_ds}

        result = ds.fill_train_and_tests()
        assert isinstance(result, tuple)
        assert len(result) == 2
