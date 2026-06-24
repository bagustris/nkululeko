# estimate.snr
"""
Module for estimating SNR (signal to noise ratio) from an audio signal.

The estimation logic (frame log-energy percentiles) now lives in the shared
``audiokit`` package — it was originally extracted from this module — so
``SNREstimator`` here subclasses ``audiokit.SNREstimator`` and only adds
nkululeko-specific extras: the matplotlib energy plot and the CLI ``main``.

This keeps a single source of truth for the SNR math (also reused by
``feats_snr`` and ``ap_snr``) while preserving this module's public API.
"""

import argparse

import audiofile
from audiokit import SNREstimator as _AudiokitSNREstimator


class SNREstimator(_AudiokitSNREstimator):
    """Estimate SNR from audio signal using log energy and energy thresholds.

    Args:
        input_data (ndarray): Input audio signal
        sample_rate (int): Sampling rate of input audio signal
        window_size (int): Window size in samples
        hop_size (int): Hop size in samples

    Returns:
        object: SNREstimator object
        estimated_snr (float): Estimated SNR in dB, extracted from SNREstimator.estimate_snr()

        Usage:
        >>> input_data, sample_rate = audiofile.read('input.wav')
        >>> snr_estimator = SNREstimator(input_data, sample_rate, window_size=320, hop_size=160)
        >>> estimated_snr, log_energies, energy_threshold_low, energy_threshold_high = snr_estimator.estimate_snr()

    The constructor, ``frame_audio``, ``calculate_log_energy``,
    ``calculate_snr`` and ``estimate_snr`` are inherited unchanged from
    ``audiokit.SNREstimator``.
    """

    def plot_energy(self, log_energies, energy_threshold_low, energy_threshold_high):
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.plot(log_energies, label="Log Energy")
        plt.axhline(
            y=energy_threshold_low,
            color="r",
            linestyle="--",
            label="Low Energy Threshold (25th Percentile)",
        )
        plt.axhline(
            y=energy_threshold_high,
            color="g",
            linestyle="--",
            label="High Energy Threshold (75th Percentile)",
        )
        plt.xlabel("Frame")
        plt.ylabel("Log Energy")
        plt.title("Log Energy and Energy Thresholds")
        plt.legend()
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Estimate SNR from audio signal",
        usage=(
            "python3 estimate_snr.py -i <input_file> -ws <window_size> -hs"
            " <hop_size> -p"
        ),
    )
    parser.add_argument(
        "-i", "--input", required=True, help="Input audio file in WAV format"
    )
    parser.add_argument(
        "-ws",
        "--window_size",
        type=int,
        default=int(0.02 * 16000),
        help="Window size in samples (default: 320)",
    )
    parser.add_argument(
        "-hs",
        "--hop_size",
        type=int,
        default=int(0.01 * 16000),
        help="Hop size in samples (default: 160)",
    )
    parser.add_argument(
        "-p",
        "--plot",
        action="store_true",
        help="Plot log energy and energy thresholds",
    )
    args = parser.parse_args()

    signal, sr = audiofile.read(args.input)
    snr_estimator = SNREstimator(signal, sr, args.window_size, args.hop_size)
    (
        estimated_snr,
        log_energies,
        energy_threshold_low,
        energy_threshold_high,
    ) = snr_estimator.estimate_snr()

    print("Estimated SNR:", estimated_snr)

    if args.plot:
        snr_estimator.plot_energy(
            log_energies, energy_threshold_low, energy_threshold_high
        )


if __name__ == "__main__":
    main()
