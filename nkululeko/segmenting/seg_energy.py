"""
seg_energy.py

Segment a dataset using audiokit's energy-VAD (a hysteresis comparator on
signal power).

Unlike the Silero / pyannote backends, this is a pure-NumPy energy detector
with no model download and no torch dependency. It detects any high-energy
events (impulsive or speech) and is therefore a good lightweight default for
non-speech audio. Select it with ``[SEGMENT] method = energy``.

It mirrors the interface of ``Silero_segmenter`` (``segment_dataframe`` returning
an audformat segmented index), including optional ``max_length`` splitting.
"""

import audformat
import audiofile
import pandas as pd
from audformat import segmented_index
from audiokit import energy_vad
from tqdm import tqdm

from nkululeko.utils.util import Util


class Energy_segmenter:
    def __init__(self, not_testing=True):
        self.no_testing = not_testing
        self.util = Util(has_config=not_testing)

    def _detect_events(self, file):
        """Read ``file[0]`` fully and return (sample_rate, [(start_s, end_s), ...]).

        Falls back to the whole file when no events cross the energy threshold,
        so every input row yields at least one segment.
        """
        signal, sampling_rate = audiofile.read(file[0], always_2d=True)
        x = signal[0]
        segments, _ = energy_vad(x, sampling_rate)
        if not segments:
            segments = [(0, len(x))]
        return sampling_rate, [
            (start / sampling_rate, end / sampling_rate) for start, end in segments
        ]

    def get_segmentation_simple(self, file):
        _, events = self._detect_events(file)
        files, starts, ends = [], [], []
        for start, end in events:
            files.append(file[0])
            starts.append(start)
            ends.append(end)
        return segmented_index(files, starts, ends)

    def get_segmentation(self, file, min_length, max_length):
        _, events = self._detect_events(file)
        files, starts, ends = [], [], []
        for start, end in events:
            new_end = end
            handled = False
            while end - start > max_length:
                new_end = start + max_length
                if end - new_end < min_length:
                    new_end = end
                files.append(file[0])
                starts.append(start)
                ends.append(new_end)
                start += max_length
                handled = True
            if not handled and end - start > min_length:
                files.append(file[0])
                starts.append(start)
                ends.append(end)
        if not files:
            # Nothing passed the length filters; keep the detected spans as-is
            # so the file is not silently dropped from the dataset.
            for start, end in events:
                files.append(file[0])
                starts.append(start)
                ends.append(end)
        return segmented_index(files, starts, ends)

    def segment_dataframe(self, df):
        dfs = []
        max_length = eval(self.util.config_val("SEGMENT", "max_length", "False"))
        min_length = 2
        if max_length:
            if self.no_testing:
                min_length = float(self.util.config_val("SEGMENT", "min_length", 2))
            self.util.debug(f"segmenting with max length: {max_length + min_length}")
        for file, values in tqdm(df.iterrows(), total=len(df)):
            if max_length:
                index = self.get_segmentation(file, min_length, max_length)
            else:
                index = self.get_segmentation_simple(file)
            dfs.append(
                pd.DataFrame(
                    values.to_dict(),
                    index,
                )
            )
        return audformat.utils.concat(dfs)
