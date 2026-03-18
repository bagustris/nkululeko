"""Tests for extract_audio_segments in nkululeko/segment.py."""

import os
import tempfile
from datetime import timedelta
from unittest.mock import MagicMock, call, patch

import numpy as np
import pandas as pd
import pytest

from nkululeko.segment import extract_audio_segments


def _make_seg_index(entries):
    """Build an audformat-style MultiIndex from (file, start_s, end_s) tuples."""
    files = [e[0] for e in entries]
    starts = [timedelta(seconds=e[1]) for e in entries]
    ends = [timedelta(seconds=e[2]) for e in entries]
    return pd.MultiIndex.from_arrays([files, starts, ends], names=["file", "start", "end"])


def _make_df_seg(entries):
    idx = _make_seg_index(entries)
    return pd.DataFrame(index=idx)


def _make_util(config_overrides=None):
    """Return a mock Util that delegates config_val to a dict."""
    defaults = {"audio_dir": "segments", "audio_format": "wav"}
    if config_overrides:
        defaults.update(config_overrides)

    util = MagicMock()
    util.config_val.side_effect = lambda section, key, default: defaults.get(key, default)
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

            with patch("nkululeko.segment.audeer.mkdir") as mock_mkdir, patch(
                "nkululeko.segment.audiofile.read", return_value=(signal, 16000)
            ), patch("nkululeko.segment.audiofile.write"):
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

            with patch("nkululeko.segment.audeer.mkdir"), patch(
                "nkululeko.segment.audiofile.read", return_value=(signal, 16000)
            ) as mock_read, patch(
                "nkululeko.segment.audiofile.write"
            ) as mock_write:
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

            with patch("nkululeko.segment.audeer.mkdir"), patch(
                "nkululeko.segment.audiofile.read", return_value=(signal, 16000)
            ), patch("nkululeko.segment.audiofile.write") as mock_write:
                extract_audio_segments(df_seg, tmpdir, util)
                written_path = mock_write.call_args[0][0]
                assert "speech_segment_000_1.0-3.5.wav" in written_path

    def test_relative_audio_dir_resolved_against_data_dir(self):
        df_seg = _make_df_seg([("/audio/a.wav", 0.0, 1.0)])
        util = _make_util({"audio_dir": "mysegments", "audio_format": "wav"})
        signal = np.zeros((1, 8000))

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("nkululeko.segment.audeer.mkdir") as mock_mkdir, patch(
                "nkululeko.segment.audiofile.read", return_value=(signal, 16000)
            ), patch("nkululeko.segment.audiofile.write"):
                extract_audio_segments(df_seg, tmpdir, util)
                expected_dir = os.path.join(tmpdir, "mysegments")
                mock_mkdir.assert_called_once_with(expected_dir)

    def test_absolute_audio_dir_not_joined_to_data_dir(self):
        with tempfile.TemporaryDirectory() as abs_dir:
            df_seg = _make_df_seg([("/audio/a.wav", 0.0, 1.0)])
            util = _make_util({"audio_dir": abs_dir, "audio_format": "wav"})
            signal = np.zeros((1, 8000))

            with patch("nkululeko.segment.audeer.mkdir") as mock_mkdir, patch(
                "nkululeko.segment.audiofile.read", return_value=(signal, 16000)
            ), patch("nkululeko.segment.audiofile.write"):
                extract_audio_segments(df_seg, "/some/data_dir", util)
                mock_mkdir.assert_called_once_with(abs_dir)

    def test_non_wav_format_in_output_name(self):
        df_seg = _make_df_seg([("/audio/clip.wav", 0.5, 2.0)])
        util = _make_util({"audio_dir": "/out", "audio_format": "flac"})
        signal = np.zeros((1, 8000))

        with patch("nkululeko.segment.audeer.mkdir"), patch(
            "nkululeko.segment.audiofile.read", return_value=(signal, 16000)
        ), patch("nkululeko.segment.audiofile.write") as mock_write:
            extract_audio_segments(df_seg, "/data", util)
            written_path = mock_write.call_args[0][0]
            assert written_path.endswith(".flac")


class TestExtractAudioSegmentsEdgeCases:
    def test_zero_duration_segment_skipped(self):
        entries = [
            ("/audio/a.wav", 1.0, 1.0),  # zero duration
            ("/audio/b.wav", 0.0, 2.0),  # valid
        ]
        df_seg = _make_df_seg(entries)
        util = _make_util()
        signal = np.zeros((1, 8000))

        with patch("nkululeko.segment.audeer.mkdir"), patch(
            "nkululeko.segment.audiofile.read", return_value=(signal, 16000)
        ) as mock_read, patch("nkululeko.segment.audiofile.write") as mock_write:
            extract_audio_segments(df_seg, "/data", util)
            assert mock_write.call_count == 1
            assert mock_read.call_count == 1

    def test_negative_duration_segment_skipped(self):
        df_seg = _make_df_seg([("/audio/a.wav", 3.0, 1.0)])  # end < start
        util = _make_util()

        with patch("nkululeko.segment.audeer.mkdir"), patch(
            "nkululeko.segment.audiofile.read"
        ) as mock_read, patch("nkululeko.segment.audiofile.write") as mock_write:
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

        with patch("nkululeko.segment.audeer.mkdir"), patch(
            "nkululeko.segment.audiofile.read", side_effect=side_effect
        ), patch("nkululeko.segment.audiofile.write") as mock_write:
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

        with patch("nkululeko.segment.audeer.mkdir"), patch(
            "nkululeko.segment.audiofile.read", side_effect=side_effect
        ), patch("nkululeko.segment.audiofile.write") as mock_write:
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

        with patch("nkululeko.segment.audeer.mkdir"), patch(
            "nkululeko.segment.audiofile.read", return_value=(signal, 16000)
        ), patch("nkululeko.segment.audiofile.write", side_effect=write_side_effect):
            extract_audio_segments(df_seg, "/data", util)
            assert len(write_calls) == 2  # both attempted despite first failure

    def test_write_permissionerror_does_not_abort(self):
        df_seg = _make_df_seg([("/audio/a.wav", 0.0, 1.0)])
        util = _make_util()
        signal = np.zeros((1, 8000))

        with patch("nkululeko.segment.audeer.mkdir"), patch(
            "nkululeko.segment.audiofile.read", return_value=(signal, 16000)
        ), patch(
            "nkululeko.segment.audiofile.write",
            side_effect=PermissionError("no write access"),
        ):
            # should not raise
            extract_audio_segments(df_seg, "/data", util)

    def test_empty_dataframe_no_writes(self):
        df_seg = pd.DataFrame(
            index=pd.MultiIndex.from_tuples([], names=["file", "start", "end"])
        )
        util = _make_util()

        with patch("nkululeko.segment.audeer.mkdir"), patch(
            "nkululeko.segment.audiofile.write"
        ) as mock_write:
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

        with patch("nkululeko.segment.audeer.mkdir"), patch(
            "nkululeko.segment.audiofile.read", return_value=(signal, 16000)
        ), patch("nkululeko.segment.audiofile.write", side_effect=capture_write):
            extract_audio_segments(df_seg, "/data", util)

        assert any("000" in p for p in written_paths)
        assert any("001" in p for p in written_paths)
        assert any("002" in p for p in written_paths)
