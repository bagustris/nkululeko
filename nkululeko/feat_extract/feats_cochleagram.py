"""Cochleagram feature extraction helpers."""

import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
from tqdm import tqdm

import nkululeko.glob_conf as glob_conf
from nkululeko.constants import SAMPLING_RATE
from nkululeko.feat_extract.feats_audio import read_indexed_audio, series_to_float_df
from nkululeko.feat_extract.featureset import Featureset

try:
    import pycochleagram.cochleagram as cgram

    _PYCOCHLEAGRAM_AVAILABLE = True
except (ImportError, AttributeError, OSError):
    cgram = None
    _PYCOCHLEAGRAM_AVAILABLE = False

# Lower priority for worker processes so extraction yields CPU to other
# users' jobs on shared machines instead of competing with them head-on.
_WORKER_NICENESS = 10

_worker_extractor = None


def _init_cochleagram_worker(extractor_kwargs):
    global _worker_extractor
    try:
        os.nice(_WORKER_NICENESS)
    except OSError:
        pass
    _worker_extractor = CochleagramFeatureExtractor(**extractor_kwargs)


def _cochleagram_worker(row_index):
    try:
        signal, _ = read_indexed_audio(row_index, _worker_extractor.sample_rate)
        return row_index, _worker_extractor.extract(signal[0]), None
    except Exception as e:
        return row_index, None, str(e)


N_BANDS = 40
LOW_LIM = 50
HI_LIM = 8000
SAMPLE_FACTOR = 2
NONLINEARITY = "db"
DOWNSAMPLE = None
FFT_MODE = "auto"
STRICT = True


def _optional_int(value):
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in {"", "none", "false"}:
        return None
    return int(text)


def _optional_str(value):
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in {"", "none", "false"}:
        return None
    return text


class CochleagramFeatureExtractor:
    """Extract summary statistics from cochleagram envelopes."""

    _warned = False

    def __init__(
        self,
        sample_rate,
        n_bands=N_BANDS,
        low_lim=LOW_LIM,
        hi_lim=HI_LIM,
        sample_factor=SAMPLE_FACTOR,
        downsample=DOWNSAMPLE,
        nonlinearity=NONLINEARITY,
        fft_mode=FFT_MODE,
        padding_size=None,
        strict=STRICT,
    ):
        self.sample_rate = sample_rate
        self.n_bands = n_bands
        self.low_lim = low_lim
        self.hi_lim = hi_lim
        self.sample_factor = sample_factor
        self.downsample = downsample
        self.nonlinearity = nonlinearity
        self.fft_mode = fft_mode
        self.padding_size = padding_size
        self.strict = strict
        self.available = _PYCOCHLEAGRAM_AVAILABLE
        self.warning = (
            "WARNING: pycochleagram not installed, skipping cochleagram features. "
            "Install from source: git clone https://github.com/mcdermottLab/pycochleagram "
            "&& cd pycochleagram && python setup.py install"
        )

    def warn_unavailable(self):
        if not self._warned:
            print(self.warning)
            self._warned = True

    def extract(self, signal_tensor):
        """Return cochleagram mean/std features for a mono signal tensor."""
        if not self.available:
            self.warn_unavailable()
            return {}

        if hasattr(signal_tensor, "cpu"):
            signal_np = signal_tensor.cpu().numpy().astype(np.float32)
        else:
            signal_np = np.asarray(signal_tensor, dtype=np.float32)
        signal_np = signal_np.reshape(-1)

        coch = cgram.human_cochleagram(
            signal_np,
            sr=self.sample_rate,
            n=self.n_bands,
            low_lim=self.low_lim,
            hi_lim=self.hi_lim,
            sample_factor=self.sample_factor,
            padding_size=self.padding_size,
            downsample=self.downsample,
            nonlinearity=self.nonlinearity,
            fft_mode=self.fft_mode,
            ret_mode="envs",
            strict=self.strict,
        )
        coch_np = np.asarray(coch, dtype=np.float32)
        if coch_np.ndim == 1:
            coch_np = coch_np[np.newaxis, :]
        elif coch_np.ndim > 2:
            coch_np = np.squeeze(coch_np)
            if coch_np.ndim == 1:
                coch_np = coch_np[np.newaxis, :]

        emb = {}
        for i in range(coch_np.shape[0]):
            band = coch_np[i]
            emb[f"cochleagram_{i}_mean"] = float(np.mean(band))
            emb[f"cochleagram_{i}_std"] = float(np.std(band))
        return emb


class CochleagramSet(Featureset):
    """Top-level feature set for `[FEATS] type = ['cochleagram']`."""

    def __init__(self, name, data_df, feats_type):
        super().__init__(name, data_df, feats_type)
        self.sample_rate = int(
            self.util.config_val("FEATS", "cochleagram.sample_rate", SAMPLING_RATE)
        )
        default_hi_lim = min(HI_LIM, max(1, self.sample_rate // 2))
        self.n_bands = int(
            self.util.config_val("FEATS", "cochleagram.n_bands", N_BANDS)
        )
        self.low_lim = int(
            self.util.config_val("FEATS", "cochleagram.low_lim", LOW_LIM)
        )
        self.hi_lim = int(
            self.util.config_val("FEATS", "cochleagram.hi_lim", default_hi_lim)
        )
        self.sample_factor = int(
            self.util.config_val("FEATS", "cochleagram.sample_factor", SAMPLE_FACTOR)
        )
        self.downsample = _optional_str(
            self.util.config_val("FEATS", "cochleagram.downsample", DOWNSAMPLE)
        )
        self.nonlinearity = _optional_str(
            self.util.config_val("FEATS", "cochleagram.nonlinearity", NONLINEARITY)
        )
        self.fft_mode = _optional_str(
            self.util.config_val("FEATS", "cochleagram.fft_mode", FFT_MODE)
        ) or FFT_MODE
        self.padding_size = _optional_int(
            self.util.config_val("FEATS", "cochleagram.padding_size", None)
        )
        self.strict = (
            str(self.util.config_val("FEATS", "cochleagram.strict", STRICT))
            .strip()
            .lower()
            == "true"
        )
        self.extractor = CochleagramFeatureExtractor(
            sample_rate=self.sample_rate,
            n_bands=self.n_bands,
            low_lim=self.low_lim,
            hi_lim=self.hi_lim,
            sample_factor=self.sample_factor,
            downsample=self.downsample,
            nonlinearity=self.nonlinearity,
            fft_mode=self.fft_mode,
            padding_size=self.padding_size,
            strict=self.strict,
        )

    def extract(self):
        """Extract cochleagram features or load them from disk if present."""
        store = self.util.get_path("store")
        store_format = self.util.config_val("FEATS", "store_format", "pkl")
        storage = f"{store}{self.name}_v2.{store_format}"
        extract = self.util.config_val("FEATS", "needs_feature_extraction", False)
        no_reuse = eval(self.util.config_val("FEATS", "no_reuse", "False"))
        if extract or no_reuse or not os.path.isfile(storage):
            self.util.debug("extracting cochleagram, this might take a while...")
            self.df = self._extract_index(self.data_df.index)
            self.util.write_store(self.df, storage, store_format)
            try:
                glob_conf.config["DATA"]["needs_feature_extraction"] = "false"
            except KeyError:
                pass
        else:
            self.util.debug("reusing extracted cochleagram values")
            self.df = self.util.get_store(storage, store_format)
            if self.df.isnull().values.any():
                nanrows = self.df.columns[self.df.isna().any()].tolist()
                print(nanrows)
                self.util.error(
                    f"got nan: {self.df.shape} {self.df.isnull().sum().sum()}"
                )

    def extract_sample(self, signal, sr):
        if not self.extractor.available:
            self.extractor.warn_unavailable()
            return pd.DataFrame([{}]).to_numpy()
        feats = self.extractor.extract(signal)
        return pd.DataFrame([feats]).astype(float).to_numpy()

    def _extract_index(self, file_index):
        if not self.extractor.available:
            self.extractor.warn_unavailable()
            return pd.DataFrame(index=file_index)
        emb_series = pd.Series(index=file_index, dtype=object)
        skipped = 0
        row_list = file_index.to_list()
        n_jobs = max(1, min(int(getattr(self, "n_jobs", 1)), os.cpu_count() or 1))

        if n_jobs > 1 and len(row_list) > 1:
            extractor_kwargs = dict(
                sample_rate=self.sample_rate,
                n_bands=self.n_bands,
                low_lim=self.low_lim,
                hi_lim=self.hi_lim,
                sample_factor=self.sample_factor,
                downsample=self.downsample,
                nonlinearity=self.nonlinearity,
                fft_mode=self.fft_mode,
                padding_size=self.padding_size,
                strict=self.strict,
            )
            with Pool(
                processes=n_jobs,
                initializer=_init_cochleagram_worker,
                initargs=(extractor_kwargs,),
            ) as pool:
                for row_index, emb, err in tqdm(
                    pool.imap(_cochleagram_worker, row_list), total=len(row_list)
                ):
                    if err is not None:
                        print(f"WARNING: featureset: skipping {row_index}: {err}")
                        skipped += 1
                    else:
                        emb_series[row_index] = emb
        else:
            for row_index in row_list:
                try:
                    signal, _ = read_indexed_audio(row_index, self.sample_rate)
                    emb_series[row_index] = self.extractor.extract(signal[0])
                except Exception as e:
                    print(f"WARNING: featureset: skipping {row_index}: {e}")
                    skipped += 1
        if skipped:
            print(
                f"WARNING: featureset: skipped {skipped} files that failed to load or extract cochleagram features"
            )
        return series_to_float_df(emb_series)