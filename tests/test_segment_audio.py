"""Tests for extract_audio_segments and CLI argument overrides in nkululeko/segment.py."""

import configparser
import os
import tempfile
from datetime import timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from nkululeko.segment import extract_audio_segments, _run_file_mode


def _make_seg_index(entries):
    """Build an audformat-style MultiIndex from (file, start_s, end_s) tuples."""
    files = [e[0] for e in entries]
    starts = [timedelta(seconds=e[1]) for e in entries]
    ends = [timedelta(seconds=e[2]) for e in entries]
    return pd.MultiIndex.from_arrays(
        [files, starts, ends], names=["file", "start", "end"]
    )


def _make_df_seg(entries):
    idx = _make_seg_index(entries)
    return pd.DataFrame(index=idx)


def _make_util(config_overrides=None):
    """Return a mock Util that delegates config_val to a dict."""
    defaults = {"audio_dir": "segments", "audio_format": "wav"}
    if config_overrides:
        defaults.update(config_overrides)

    util = MagicMock()
    util.config_val.side_effect = lambda section, key, default: defaults.get(
        key, default
    )
    return util


class TestExtractAudioSegmentsBasic:
    def test_output_directory_created(self):
        df_seg = _make_df_seg([("/audio/file1.wav", 0.0, 2.5)])
        util = _make_util()
        signal = np.zeros((1, 16000))

        with tempfile.TemporaryDirectory() as tmpdir:
            util.config_val.side_effect = lambda s, k, d: {
                "audio_dir": tmpdir,
                "audio_format": "wav",
            }.get(k, d)

            with (
                patch("nkululeko.segment.audeer.mkdir") as mock_mkdir,
                patch("nkululeko.segment.audiofile.read", return_value=(signal, 16000)),
                patch("nkululeko.segment.audiofile.write"),
            ):
                extract_audio_segments(df_seg, tmpdir, util)
                mock_mkdir.assert_called_once_with(tmpdir)

    def test_write_called_for_each_valid_segment(self):
        entries = [
            ("/audio/a.wav", 0.0, 1.0),
            ("/audio/b.wav", 2.0, 4.0),
        ]
        df_seg = _make_df_seg(entries)
        util = _make_util()
        signal = np.zeros((1, 8000))

        with tempfile.TemporaryDirectory() as tmpdir:
            util.config_val.side_effect = lambda s, k, d: {
                "audio_dir": tmpdir,
                "audio_format": "wav",
            }.get(k, d)

            with (
                patch("nkululeko.segment.audeer.mkdir"),
                patch("nkululeko.segment.audiofile.read", return_value=(signal, 16000)),
                patch("nkululeko.segment.audiofile.write") as mock_write,
            ):
                extract_audio_segments(df_seg, tmpdir, util)
                assert mock_write.call_count == 2

    def test_output_filename_pattern(self):
        df_seg = _make_df_seg([("/audio/speech.wav", 1.0, 3.5)])
        util = _make_util()
        signal = np.zeros((1, 8000))

        with tempfile.TemporaryDirectory() as tmpdir:
            util.config_val.side_effect = lambda s, k, d: {
                "audio_dir": tmpdir,
                "audio_format": "wav",
            }.get(k, d)

            with (
                patch("nkululeko.segment.audeer.mkdir"),
                patch("nkululeko.segment.audiofile.read", return_value=(signal, 16000)),
                patch("nkululeko.segment.audiofile.write") as mock_write,
            ):
                extract_audio_segments(df_seg, tmpdir, util)
                written_path = mock_write.call_args[0][0]
                assert "speech_segment_000_1.0-3.5.wav" in written_path

    def test_relative_audio_dir_resolved_against_data_dir(self):
        df_seg = _make_df_seg([("/audio/a.wav", 0.0, 1.0)])
        util = _make_util({"audio_dir": "mysegments", "audio_format": "wav"})
        signal = np.zeros((1, 8000))

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch("nkululeko.segment.audeer.mkdir") as mock_mkdir,
                patch("nkululeko.segment.audiofile.read", return_value=(signal, 16000)),
                patch("nkululeko.segment.audiofile.write"),
            ):
                extract_audio_segments(df_seg, tmpdir, util)
                expected_dir = os.path.join(tmpdir, "mysegments")
                mock_mkdir.assert_called_once_with(expected_dir)

    def test_absolute_audio_dir_not_joined_to_data_dir(self):
        with tempfile.TemporaryDirectory() as abs_dir:
            df_seg = _make_df_seg([("/audio/a.wav", 0.0, 1.0)])
            util = _make_util({"audio_dir": abs_dir, "audio_format": "wav"})
            signal = np.zeros((1, 8000))

            with (
                patch("nkululeko.segment.audeer.mkdir") as mock_mkdir,
                patch("nkululeko.segment.audiofile.read", return_value=(signal, 16000)),
                patch("nkululeko.segment.audiofile.write"),
            ):
                extract_audio_segments(df_seg, "/some/data_dir", util)
                mock_mkdir.assert_called_once_with(abs_dir)

    def test_non_wav_format_in_output_name(self):
        df_seg = _make_df_seg([("/audio/clip.wav", 0.5, 2.0)])
        util = _make_util({"audio_dir": "/out", "audio_format": "flac"})
        signal = np.zeros((1, 8000))

        with (
            patch("nkululeko.segment.audeer.mkdir"),
            patch("nkululeko.segment.audiofile.read", return_value=(signal, 16000)),
            patch("nkululeko.segment.audiofile.write") as mock_write,
        ):
            extract_audio_segments(df_seg, "/data", util)
            written_path = mock_write.call_args[0][0]
            assert written_path.endswith(".flac")


class TestSamplingRateExport:
    """Tests for the optional sampling_rate resampling on audio export."""

    def test_no_sampling_rate_preserves_original(self):
        """Omitting sampling_rate writes the signal at the original rate."""
        df_seg = _make_df_seg([("/audio/a.wav", 0.0, 1.0)])
        util = _make_util({"audio_dir": "/out", "audio_format": "wav"})
        signal = np.zeros((1, 16000))

        with (
            patch("nkululeko.segment.audeer.mkdir"),
            patch("nkululeko.segment.audiofile.read", return_value=(signal, 44100)),
            patch("nkululeko.segment.audiofile.write") as mock_write,
        ):
            extract_audio_segments(df_seg, "/data", util)
            _, written_sr = mock_write.call_args[0][1], mock_write.call_args[0][2]
            assert written_sr == 44100

    def test_sampling_rate_same_as_source_skips_resample(self):
        """When target_sr equals source SR, torchaudio.functional.resample is not called."""
        df_seg = _make_df_seg([("/audio/a.wav", 0.0, 1.0)])
        util = _make_util(
            {"audio_dir": "/out", "audio_format": "wav", "sampling_rate": "16000"}
        )
        signal = np.zeros((1, 16000))

        with (
            patch("nkululeko.segment.audeer.mkdir"),
            patch("nkululeko.segment.audiofile.read", return_value=(signal, 16000)),
            patch("nkululeko.segment.audiofile.write") as mock_write,
        ):
            mock_torchaudio = MagicMock()
            with patch.dict(
                "sys.modules", {"torchaudio": mock_torchaudio, "torch": MagicMock()}
            ):
                extract_audio_segments(df_seg, "/data", util)
            mock_torchaudio.functional.resample.assert_not_called()
            assert mock_write.call_args[0][2] == 16000

    def test_sampling_rate_triggers_resample_and_writes_target_sr(self):
        """When target_sr differs from source SR, written SR matches target."""
        df_seg = _make_df_seg([("/audio/a.wav", 0.0, 1.0)])
        util = _make_util(
            {"audio_dir": "/out", "audio_format": "wav", "sampling_rate": "16000"}
        )
        source_signal = np.zeros((1, 44100), dtype=np.float32)
        resampled_signal = np.zeros((1, 16000), dtype=np.float32)

        mock_torch = MagicMock()
        mock_torch.from_numpy.return_value = MagicMock()
        mock_torchaudio = MagicMock()
        # transforms.Resample(orig, target)(signal).numpy() -> resampled_signal
        mock_torchaudio.transforms.Resample.return_value.return_value.numpy.return_value = resampled_signal

        with (
            patch("nkululeko.segment.audeer.mkdir"),
            patch(
                "nkululeko.segment.audiofile.read", return_value=(source_signal, 44100)
            ),
            patch("nkululeko.segment.audiofile.write") as mock_write,
            patch.dict(
                "sys.modules", {"torchaudio": mock_torchaudio, "torch": mock_torch}
            ),
        ):
            extract_audio_segments(df_seg, "/data", util)

        mock_torchaudio.transforms.Resample.assert_called_once_with(44100, 16000)
        assert mock_write.call_args[0][2] == 16000

    def test_sampling_rate_ini_override(self, tmp_path):
        """[SEGMENT] sampling_rate in INI is forwarded to config before Experiment loads."""
        import configparser
        from nkululeko.segment import main

        config = configparser.ConfigParser()
        config["EXP"] = {"name": "test", "root": str(tmp_path)}
        config["DATA"] = {"databases": "['db']", "target": "emotion"}
        config["FEATS"] = {"type": "['os']"}
        config["MODEL"] = {"type": "xgb"}
        config["SEGMENT"] = {}
        ini_path = str(tmp_path / "exp.ini")
        with open(ini_path, "w") as f:
            config.write(f)

        captured = {}

        def fake_experiment(cfg):
            captured["sampling_rate"] = cfg.get(
                "SEGMENT", "sampling_rate", fallback=None
            )
            raise SystemExit(0)

        with (
            patch("nkululeko.segment.Experiment", side_effect=fake_experiment),
            patch(
                "sys.argv",
                ["segment", "--config", ini_path, "--sampling_rate", "16000"],
            ),
        ):
            with pytest.raises(SystemExit):
                main()

        assert captured["sampling_rate"] == "16000"

    def test_sampling_rate_file_mode_injected(self, tmp_path):
        """--sampling_rate in file mode is stored in glob_conf.config."""
        import nkululeko.glob_conf as glob_conf
        from nkululeko.segment import _run_file_mode

        dummy_wav = str(tmp_path / "a.wav")
        open(dummy_wav, "wb").close()

        args = MagicMock()
        args.file = dummy_wav
        args.max_length = None
        args.min_length = None
        args.output_audio = False
        args.sampling_rate = 16000

        seg_df = _make_df_seg([(dummy_wav, 0.0, 2.0)])
        mock_module = MagicMock()
        mock_module.Silero_segmenter.return_value.segment_dataframe.return_value = (
            seg_df
        )
        idx = pd.MultiIndex.from_tuples(
            [(dummy_wav, timedelta(0), timedelta(seconds=5.0))],
            names=["file", "start", "end"],
        )
        with (
            patch.dict("sys.modules", {"nkululeko.segmenting.seg_silero": mock_module}),
            patch(
                "nkululeko.segment.audformat.utils.to_segmented_index", return_value=idx
            ),
            patch("nkululeko.segment.extract_audio_segments"),
        ):
            _run_file_mode(args)

        assert glob_conf.config.get("SEGMENT", "sampling_rate") == "16000"


class TestExtractAudioSegmentsEdgeCases:
    def test_zero_duration_segment_skipped(self):
        entries = [
            ("/audio/a.wav", 1.0, 1.0),  # zero duration
            ("/audio/b.wav", 0.0, 2.0),  # valid
        ]
        df_seg = _make_df_seg(entries)
        util = _make_util()
        signal = np.zeros((1, 8000))

        with (
            patch("nkululeko.segment.audeer.mkdir"),
            patch(
                "nkululeko.segment.audiofile.read", return_value=(signal, 16000)
            ) as mock_read,
            patch("nkululeko.segment.audiofile.write") as mock_write,
        ):
            extract_audio_segments(df_seg, "/data", util)
            assert mock_write.call_count == 1
            assert mock_read.call_count == 1

    def test_negative_duration_segment_skipped(self):
        df_seg = _make_df_seg([("/audio/a.wav", 3.0, 1.0)])  # end < start
        util = _make_util()

        with (
            patch("nkululeko.segment.audeer.mkdir"),
            patch("nkululeko.segment.audiofile.read") as mock_read,
            patch("nkululeko.segment.audiofile.write") as mock_write,
        ):
            extract_audio_segments(df_seg, "/data", util)
            mock_read.assert_not_called()
            mock_write.assert_not_called()

    def test_read_oserror_skips_segment_continues(self):
        entries = [
            ("/audio/bad.wav", 0.0, 1.0),
            ("/audio/good.wav", 0.0, 1.0),
        ]
        df_seg = _make_df_seg(entries)
        util = _make_util()
        signal = np.zeros((1, 8000))

        def side_effect(path, **kwargs):
            if "bad" in path:
                raise OSError("file not found")
            return signal, 16000

        with (
            patch("nkululeko.segment.audeer.mkdir"),
            patch("nkululeko.segment.audiofile.read", side_effect=side_effect),
            patch("nkululeko.segment.audiofile.write") as mock_write,
        ):
            extract_audio_segments(df_seg, "/data", util)
            assert mock_write.call_count == 1

    def test_read_runtimeerror_skips_segment_continues(self):
        entries = [
            ("/audio/corrupt.wav", 0.0, 1.0),
            ("/audio/ok.wav", 0.5, 2.0),
        ]
        df_seg = _make_df_seg(entries)
        util = _make_util()
        signal = np.zeros((1, 8000))

        def side_effect(path, **kwargs):
            if "corrupt" in path:
                raise RuntimeError("decoding error")
            return signal, 16000

        with (
            patch("nkululeko.segment.audeer.mkdir"),
            patch("nkululeko.segment.audiofile.read", side_effect=side_effect),
            patch("nkululeko.segment.audiofile.write") as mock_write,
        ):
            extract_audio_segments(df_seg, "/data", util)
            assert mock_write.call_count == 1

    def test_write_oserror_does_not_abort(self):
        entries = [
            ("/audio/a.wav", 0.0, 1.0),
            ("/audio/b.wav", 1.0, 2.0),
        ]
        df_seg = _make_df_seg(entries)
        util = _make_util()
        signal = np.zeros((1, 8000))
        write_calls = []

        def write_side_effect(path, sig, sr):
            write_calls.append(path)
            if "000" in path:
                raise OSError("disk full")

        with (
            patch("nkululeko.segment.audeer.mkdir"),
            patch("nkululeko.segment.audiofile.read", return_value=(signal, 16000)),
            patch("nkululeko.segment.audiofile.write", side_effect=write_side_effect),
        ):
            extract_audio_segments(df_seg, "/data", util)
            assert len(write_calls) == 2  # both attempted despite first failure

    def test_write_permissionerror_does_not_abort(self):
        df_seg = _make_df_seg([("/audio/a.wav", 0.0, 1.0)])
        util = _make_util()
        signal = np.zeros((1, 8000))

        with (
            patch("nkululeko.segment.audeer.mkdir"),
            patch("nkululeko.segment.audiofile.read", return_value=(signal, 16000)),
            patch(
                "nkululeko.segment.audiofile.write",
                side_effect=PermissionError("no write access"),
            ),
        ):
            # should not raise
            extract_audio_segments(df_seg, "/data", util)

    def test_empty_dataframe_no_writes(self):
        df_seg = pd.DataFrame(
            index=pd.MultiIndex.from_tuples([], names=["file", "start", "end"])
        )
        util = _make_util()

        with (
            patch("nkululeko.segment.audeer.mkdir"),
            patch("nkululeko.segment.audiofile.write") as mock_write,
        ):
            extract_audio_segments(df_seg, "/data", util)
            mock_write.assert_not_called()

    def test_segment_index_counter_increments(self):
        entries = [
            ("/audio/a.wav", 0.0, 1.0),
            ("/audio/b.wav", 2.0, 3.0),
            ("/audio/c.wav", 4.0, 5.0),
        ]
        df_seg = _make_df_seg(entries)
        util = _make_util()
        signal = np.zeros((1, 8000))
        written_paths = []

        def capture_write(path, sig, sr):
            written_paths.append(os.path.basename(path))

        with (
            patch("nkululeko.segment.audeer.mkdir"),
            patch("nkululeko.segment.audiofile.read", return_value=(signal, 16000)),
            patch("nkululeko.segment.audiofile.write", side_effect=capture_write),
        ):
            extract_audio_segments(df_seg, "/data", util)

        assert any("000" in p for p in written_paths)
        assert any("001" in p for p in written_paths)
        assert any("002" in p for p in written_paths)


class TestCLIArgumentOverrides:
    """Test that CLI args are injected into config before Experiment is created."""

    def _make_minimal_ini(self, tmp_path, extra_segment=None):
        """Write a minimal INI file and return its path."""
        config = configparser.ConfigParser()
        config["EXP"] = {"name": "test", "root": str(tmp_path)}
        config["DATA"] = {"databases": "['db']", "target": "emotion"}
        config["FEATS"] = {"type": "['os']"}
        config["MODEL"] = {"type": "xgb"}
        config["SEGMENT"] = extra_segment or {}
        ini_path = str(tmp_path / "exp.ini")
        with open(ini_path, "w") as f:
            config.write(f)
        return ini_path

    def test_max_length_arg_overrides_ini(self, tmp_path):
        """--max_length sets SEGMENT.max_length in config before Experiment loads."""
        from nkululeko.segment import main

        ini = self._make_minimal_ini(tmp_path, {"max_length": "10"})
        captured_config = {}

        def fake_experiment(config):
            captured_config["max_length"] = config.get(
                "SEGMENT", "max_length", fallback=None
            )
            raise SystemExit(0)

        with (
            patch("nkululeko.segment.Experiment", side_effect=fake_experiment),
            patch("sys.argv", ["segment", "--config", ini, "--max_length", "30"]),
        ):
            with pytest.raises(SystemExit):
                main()

        assert captured_config["max_length"] == "30.0"

    def test_min_length_arg_overrides_ini(self, tmp_path):
        """--min_length sets SEGMENT.min_length in config."""
        from nkululeko.segment import main

        ini = self._make_minimal_ini(tmp_path, {"min_length": "2"})
        captured_config = {}

        def fake_experiment(config):
            captured_config["min_length"] = config.get(
                "SEGMENT", "min_length", fallback=None
            )
            raise SystemExit(0)

        with (
            patch("nkululeko.segment.Experiment", side_effect=fake_experiment),
            patch("sys.argv", ["segment", "--config", ini, "--min_length", "5"]),
        ):
            with pytest.raises(SystemExit):
                main()

        assert captured_config["min_length"] == "5.0"

    def test_output_audio_flag_sets_true_in_config(self, tmp_path):
        """--output_audio sets SEGMENT.output_audio = True in config."""
        from nkululeko.segment import main

        ini = self._make_minimal_ini(tmp_path)
        captured_config = {}

        def fake_experiment(config):
            captured_config["output_audio"] = config.get(
                "SEGMENT", "output_audio", fallback="False"
            )
            raise SystemExit(0)

        with (
            patch("nkululeko.segment.Experiment", side_effect=fake_experiment),
            patch("sys.argv", ["segment", "--config", ini, "--output_audio"]),
        ):
            with pytest.raises(SystemExit):
                main()

        assert captured_config["output_audio"] == "True"

    def test_output_audio_absent_does_not_override_ini(self, tmp_path):
        """Omitting --output_audio leaves the INI value intact."""
        from nkululeko.segment import main

        ini = self._make_minimal_ini(tmp_path, {"output_audio": "False"})
        captured_config = {}

        def fake_experiment(config):
            captured_config["output_audio"] = config.get(
                "SEGMENT", "output_audio", fallback="False"
            )
            raise SystemExit(0)

        with (
            patch("nkululeko.segment.Experiment", side_effect=fake_experiment),
            patch("sys.argv", ["segment", "--config", ini]),
        ):
            with pytest.raises(SystemExit):
                main()

        assert captured_config["output_audio"] == "False"

    def test_max_length_absent_does_not_override_ini(self, tmp_path):
        """Omitting --max_length leaves the INI value intact."""
        from nkululeko.segment import main

        ini = self._make_minimal_ini(tmp_path, {"max_length": "15"})
        captured_config = {}

        def fake_experiment(config):
            captured_config["max_length"] = config.get(
                "SEGMENT", "max_length", fallback=None
            )
            raise SystemExit(0)

        with (
            patch("nkululeko.segment.Experiment", side_effect=fake_experiment),
            patch("sys.argv", ["segment", "--config", ini]),
        ):
            with pytest.raises(SystemExit):
                main()

        assert captured_config["max_length"] == "15"

    def test_segment_section_created_if_missing_from_ini(self, tmp_path):
        """CLI args work even when [SEGMENT] section is absent from the INI."""
        from nkululeko.segment import main

        config = configparser.ConfigParser()
        config["EXP"] = {"name": "test", "root": str(tmp_path)}
        config["DATA"] = {"databases": "['db']", "target": "emotion"}
        config["FEATS"] = {"type": "['os']"}
        config["MODEL"] = {"type": "xgb"}
        # No [SEGMENT] section
        ini_path = str(tmp_path / "no_segment.ini")
        with open(ini_path, "w") as f:
            config.write(f)

        captured_config = {}

        def fake_experiment(cfg):
            captured_config["has_section"] = cfg.has_section("SEGMENT")
            captured_config["max_length"] = cfg.get(
                "SEGMENT", "max_length", fallback=None
            )
            raise SystemExit(0)

        with (
            patch("nkululeko.segment.Experiment", side_effect=fake_experiment),
            patch("sys.argv", ["segment", "--config", ini_path, "--max_length", "20"]),
        ):
            with pytest.raises(SystemExit):
                main()

        assert captured_config["has_section"] is True
        assert captured_config["max_length"] == "20.0"


class TestFileModeRouting:
    """Test --file mode: routing, config injection, and audio export."""

    def _make_args(
        self,
        file,
        max_length=None,
        min_length=None,
        output_audio=False,
        sampling_rate=None,
    ):
        args = MagicMock()
        args.file = file
        args.max_length = max_length
        args.min_length = min_length
        args.output_audio = output_audio
        args.sampling_rate = sampling_rate
        return args

    def _make_seg_df(self, file, pairs):
        """Build a segmented DataFrame from (start_s, end_s) pairs."""
        entries = [(file, s, e) for s, e in pairs]
        idx = _make_seg_index(entries)
        return pd.DataFrame({"duration": [e - s for _, s, e in entries]}, index=idx)

    def _mock_silero(self, seg_df):
        """Return a sys.modules patch that mocks Silero_segmenter in seg_silero."""
        mock_module = MagicMock()
        mock_module.Silero_segmenter.return_value.segment_dataframe.return_value = (
            seg_df
        )
        return patch.dict(
            "sys.modules", {"nkululeko.segmenting.seg_silero": mock_module}
        )

    def _mock_to_seg_index(self, file, duration=5.0):
        """Return a patch for audformat.utils.to_segmented_index to avoid real audio reads."""
        from datetime import timedelta

        idx = pd.MultiIndex.from_tuples(
            [(file, timedelta(0), timedelta(seconds=duration))],
            names=["file", "start", "end"],
        )
        return patch(
            "nkululeko.segment.audformat.utils.to_segmented_index", return_value=idx
        )

    def test_missing_file_exits(self, tmp_path):
        args = self._make_args(str(tmp_path / "nonexistent.wav"))
        with pytest.raises(SystemExit):
            _run_file_mode(args)

    def test_max_length_injected_into_glob_conf(self, tmp_path):
        """max_length from args must be visible in glob_conf.config after _run_file_mode."""
        import nkululeko.glob_conf as glob_conf

        dummy_wav = str(tmp_path / "speech.wav")
        open(dummy_wav, "wb").close()

        args = self._make_args(dummy_wav, max_length=30.0)
        seg_df = self._make_seg_df(dummy_wav, [(0.0, 2.0)])

        with (
            self._mock_silero(seg_df),
            self._mock_to_seg_index(dummy_wav),
            patch("nkululeko.segment.extract_audio_segments"),
        ):
            _run_file_mode(args)

        assert glob_conf.config.get("SEGMENT", "max_length") == "30.0"

    def test_min_length_injected_into_glob_conf(self, tmp_path):
        import nkululeko.glob_conf as glob_conf

        dummy_wav = str(tmp_path / "speech.wav")
        open(dummy_wav, "wb").close()

        args = self._make_args(dummy_wav, min_length=3.0)
        seg_df = self._make_seg_df(dummy_wav, [(0.0, 5.0)])

        with (
            self._mock_silero(seg_df),
            self._mock_to_seg_index(dummy_wav),
            patch("nkululeko.segment.extract_audio_segments"),
        ):
            _run_file_mode(args)

        assert glob_conf.config.get("SEGMENT", "min_length") == "3.0"

    def test_no_length_args_segment_section_still_created(self, tmp_path):
        """Even without length args, [SEGMENT] section must exist in glob_conf.config."""
        import nkululeko.glob_conf as glob_conf

        dummy_wav = str(tmp_path / "speech.wav")
        open(dummy_wav, "wb").close()

        args = self._make_args(dummy_wav)
        seg_df = self._make_seg_df(dummy_wav, [(0.0, 1.5)])

        with (
            self._mock_silero(seg_df),
            self._mock_to_seg_index(dummy_wav),
            patch("nkululeko.segment.extract_audio_segments"),
        ):
            _run_file_mode(args)

        assert glob_conf.config.has_section("SEGMENT")

    def test_silero_segmenter_called_with_not_testing_true(self, tmp_path):
        dummy_wav = str(tmp_path / "a.wav")
        open(dummy_wav, "wb").close()

        args = self._make_args(dummy_wav)
        seg_df = self._make_seg_df(dummy_wav, [(0.0, 2.0)])

        mock_module = MagicMock()
        mock_module.Silero_segmenter.return_value.segment_dataframe.return_value = (
            seg_df
        )
        with (
            patch.dict("sys.modules", {"nkululeko.segmenting.seg_silero": mock_module}),
            self._mock_to_seg_index(dummy_wav),
            patch("nkululeko.segment.extract_audio_segments"),
        ):
            _run_file_mode(args)
            mock_module.Silero_segmenter.assert_called_once_with(not_testing=True)

    def test_output_audio_calls_extract(self, tmp_path):
        dummy_wav = str(tmp_path / "a.wav")
        open(dummy_wav, "wb").close()

        args = self._make_args(dummy_wav, output_audio=True)
        seg_df = self._make_seg_df(dummy_wav, [(0.0, 2.0)])

        with (
            self._mock_silero(seg_df),
            self._mock_to_seg_index(dummy_wav),
            patch("nkululeko.segment.extract_audio_segments") as mock_extract,
        ):
            _run_file_mode(args)
            mock_extract.assert_called_once()
            assert mock_extract.call_args[0][1] == str(tmp_path)

    def test_output_audio_false_does_not_call_extract(self, tmp_path):
        dummy_wav = str(tmp_path / "a.wav")
        open(dummy_wav, "wb").close()

        args = self._make_args(dummy_wav, output_audio=False)
        seg_df = self._make_seg_df(dummy_wav, [(0.0, 2.0)])

        with (
            self._mock_silero(seg_df),
            self._mock_to_seg_index(dummy_wav),
            patch("nkululeko.segment.extract_audio_segments") as mock_extract,
        ):
            _run_file_mode(args)
            mock_extract.assert_not_called()

    def test_file_mode_via_main_dispatch(self, tmp_path):
        """main() must dispatch to _run_file_mode when --file is given."""
        from nkululeko.segment import main

        dummy_wav = str(tmp_path / "speech.wav")
        open(dummy_wav, "wb").close()

        with (
            patch("nkululeko.segment._run_file_mode") as mock_file_mode,
            patch("sys.argv", ["segment", "--file", dummy_wav]),
        ):
            main()
            mock_file_mode.assert_called_once()
            assert mock_file_mode.call_args[0][0].file == dummy_wav

    def test_no_args_exits(self, tmp_path):
        from nkululeko.segment import main

        with patch("sys.argv", ["segment"]):
            with pytest.raises(SystemExit):
                main()
