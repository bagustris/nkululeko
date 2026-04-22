"""Tests for nkululeko/runmanager.py — Runmanager class."""

import configparser
import types

import pytest

import nkululeko.glob_conf as glob_conf
from nkululeko.runmanager import Runmanager


@pytest.fixture(autouse=True)
def setup_glob_conf(tmp_path):
    config = configparser.ConfigParser()
    config["EXP"] = {
        "type": "classification",
        "name": "test_run",
        "root": str(tmp_path),
        "runs": "1",
        "epochs": "1",
        "traindevtest": "False",
    }
    config["DATA"] = {"target": "emotion", "databases": "['test_db']"}
    config["MODEL"] = {"type": "xgb", "measure": "uar"}
    config["FEATS"] = {"type": "['os']"}
    config["PLOT"] = {}
    glob_conf.init_config(config)
    yield
    glob_conf.config = None


def _make_report(test_score, run=0, epoch=0):
    """Create a minimal mock report with result.test set."""
    result = types.SimpleNamespace(test=test_score)
    report = types.SimpleNamespace(result=result, run=run, epoch=epoch)
    return report


@pytest.fixture
def runmanager(tmp_path):
    """Construct a Runmanager with empty dataframes (no actual training)."""
    import pandas as pd

    df = pd.DataFrame({"emotion": []})
    feats = pd.DataFrame()
    return Runmanager(df, df, feats, feats)


class TestSearchBestResultAscending:
    def test_picks_highest_value(self, runmanager):
        reports = [
            _make_report(0.3),
            _make_report(0.7),
            _make_report(0.5),
        ]
        best = runmanager.search_best_result(reports, "ascending")
        assert best.result.test == 0.7

    def test_returns_first_with_all_equal(self, runmanager):
        reports = [_make_report(0.5), _make_report(0.5)]
        best = runmanager.search_best_result(reports, "ascending")
        assert best.result.test == 0.5

    def test_single_report(self, runmanager):
        reports = [_make_report(0.42)]
        best = runmanager.search_best_result(reports, "ascending")
        assert best.result.test == 0.42


class TestSearchBestResultDescending:
    def test_picks_lowest_value(self, runmanager):
        reports = [
            _make_report(0.3),
            _make_report(0.7),
            _make_report(0.1),
        ]
        best = runmanager.search_best_result(reports, "descending")
        assert best.result.test == 0.1

    def test_single_report(self, runmanager):
        reports = [_make_report(0.05)]
        best = runmanager.search_best_result(reports, "descending")
        assert best.result.test == 0.05


class TestGetBestResult:
    def test_classification_uar_uses_ascending(self, runmanager):
        """UAR is higher-is-better, so get_best_result should return the max."""
        reports = [_make_report(0.2), _make_report(0.9), _make_report(0.5)]
        best = runmanager.get_best_result(reports)
        assert best.result.test == 0.9

    def test_classification_eer_uses_descending(self, runmanager):
        """EER is lower-is-better."""
        glob_conf.config["MODEL"]["measure"] = "eer"
        reports = [_make_report(0.2), _make_report(0.9), _make_report(0.05)]
        best = runmanager.get_best_result(reports)
        assert best.result.test == 0.05

    def test_regression_mse_uses_descending(self, runmanager):
        """MSE is lower-is-better."""
        glob_conf.config["EXP"]["type"] = "regression"
        glob_conf.config["MODEL"]["measure"] = "mse"
        reports = [_make_report(10.0), _make_report(0.5), _make_report(3.0)]
        best = runmanager.get_best_result(reports)
        assert best.result.test == 0.5


class TestRunmanagerInit:
    def test_stores_split3_false(self, runmanager):
        assert runmanager.split3 is False

    def test_target_from_config(self, runmanager):
        assert runmanager.target == "emotion"

    def test_split3_true_when_configured(self, tmp_path):
        import pandas as pd

        glob_conf.config["EXP"]["traindevtest"] = "True"
        df = pd.DataFrame({"emotion": []})
        feats = pd.DataFrame()
        rm = Runmanager(df, df, feats, feats)
        assert rm.split3 is True
