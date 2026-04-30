#!/usr/bin/env python3
"""Process TIMIT dataset for phoneme or dialect recognition with nkululeko.

Phoneme task: reads PHN files to create segmented CSV files (file, start, end, phoneme).
  Maps 61 TIMIT phonemes to the standard 39-class set (Lee & Hon, 1989).
  Silence and closure segments are excluded.

Dialect task: creates utterance-level CSV files (file, dialect, speaker).
  Labels are DR1–DR8 from the TIMIT directory structure.
  All 8 dialect regions are always included regardless of --dialect-regions.

Usage:
    python process_database.py [--task {phoneme,dialect,both}]
                               [--timit-dir TIMIT_DIR] [--out-dir OUT_DIR]
                               [--val-fraction VAL_FRACTION]
                               [--dialect-regions DR1 DR2 ...]

Outputs (in OUT_DIR):
    phoneme task: timit_train.csv, timit_val.csv, timit_test.csv
    dialect task: timit_dialect_train.csv, timit_dialect_val.csv, timit_dialect_test.csv
"""

import argparse
import pathlib
import random

import pandas as pd

# 61 → 39 TIMIT phoneme mapping. None = excluded (silence, closures).
PHONEME_MAP = {
    # Vowels
    "iy": "iy", "ih": "ih", "ix": "ih",
    "eh": "eh", "ae": "ae",
    "ah": "ah", "ax": "ah", "ax-h": "ah",
    "uw": "uw", "ux": "uw", "uh": "uh",
    "ao": "ao", "aa": "aa",
    "ey": "ey", "ay": "ay", "oy": "oy", "aw": "aw", "ow": "ow",
    "er": "er", "axr": "er",
    # Approximants / liquids / glides
    "l": "l", "el": "l",
    "r": "r", "w": "w", "y": "y",
    # Fricatives
    "hh": "hh", "hv": "hh",
    "s": "s", "sh": "sh", "z": "z", "zh": "sh",
    "f": "f", "th": "th", "v": "v", "dh": "dh",
    # Stops
    "b": "b", "d": "d", "g": "g",
    "p": "p", "t": "t", "k": "k",
    "dx": "dx",
    # Affricates
    "jh": "jh", "ch": "ch",
    # Nasals
    "m": "m", "em": "m",
    "n": "n", "en": "n", "nx": "n",
    "ng": "ng", "eng": "ng",
    # Excluded: silence, closures, glottal stop
    "bcl": None, "dcl": None, "gcl": None, "pcl": None, "tcl": None, "kcl": None,
    "h#": None, "pau": None, "epi": None, "q": None,
}

SR = 16000  # TIMIT sample rate

# All 8 TIMIT dialect regions with descriptions
DIALECT_LABELS = ["DR1", "DR2", "DR3", "DR4", "DR5", "DR6", "DR7", "DR8"]
DIALECT_NAMES = {
    "DR1": "New England",
    "DR2": "Northern",
    "DR3": "North Midland",
    "DR4": "South Midland",
    "DR5": "Southern",
    "DR6": "New York City",
    "DR7": "Western",
    "DR8": "Army Brat (mobile)",
}


def parse_phn(phn_path: pathlib.Path):
    """Return list of (start_sec, end_sec, phoneme_39) from a .PHN file."""
    segments = []
    with open(phn_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            start_s, end_s, raw = int(parts[0]), int(parts[1]), parts[2]
            mapped = PHONEME_MAP.get(raw)
            if mapped is None:
                continue
            segments.append((round(start_s / SR, 6), round(end_s / SR, 6), mapped))
    return segments


def main():
    parser = argparse.ArgumentParser(
        description="Process TIMIT for phoneme or dialect recognition."
    )
    parser.add_argument(
        "--task",
        choices=["phoneme", "dialect", "both"],
        default="phoneme",
        help="Task to generate CSVs for (default: phoneme)",
    )
    parser.add_argument(
        "--timit-dir",
        default="./data/timit/TIMIT",
        help="Path to the TIMIT root directory (contains TRAIN/ and TEST/)",
    )
    parser.add_argument(
        "--out-dir",
        default="./data/timit",
        help="Directory to write the output CSV files",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Fraction of TRAIN speakers to use for validation (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val speaker split",
    )
    parser.add_argument(
        "--dialect-regions",
        nargs="+",
        default=None,
        metavar="DR",
        help="Limit phoneme task to specific dialect region(s), e.g. DR1 DR2. "
             "Default: all 8 regions. Ignored for the dialect task (always all 8).",
    )
    args = parser.parse_args()

    timit_dir = pathlib.Path(args.timit_dir)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_dir = timit_dir / "TRAIN"
    test_dir = timit_dir / "TEST"

    def make_rel(wav_file: pathlib.Path) -> str:
        """Return path relative to out_dir, following the symlink name."""
        try:
            return wav_file.relative_to(out_dir).as_posix()
        except ValueError:
            timit_name = timit_dir.name
            parts = wav_file.parts
            try:
                idx = next(i for i, p in enumerate(parts) if p == timit_name)
                return "/".join([timit_name] + list(parts[idx + 1:]))
            except StopIteration:
                return str(wav_file)

    # ── Compute speaker-based train / val split (shared across tasks) ──────────
    do_phoneme = args.task in ("phoneme", "both")
    do_dialect = args.task in ("dialect", "both")

    # For the phoneme task we may restrict to certain DRs; dialect always uses all.
    phoneme_allowed_drs = (
        {d.upper() for d in args.dialect_regions} if args.dialect_regions else None
    )
    if do_phoneme and phoneme_allowed_drs:
        print(f"Phoneme task: limiting to dialect region(s): {sorted(phoneme_allowed_drs)}")

    # Collect all TRAIN speakers (from all DRs — shared split for both tasks)
    all_speakers = []
    for dr_dir in sorted(train_dir.iterdir()):
        if dr_dir.is_dir():
            for spk_dir in sorted(dr_dir.iterdir()):
                if spk_dir.is_dir():
                    all_speakers.append(spk_dir)

    rng = random.Random(args.seed)
    rng.shuffle(all_speakers)
    n_val = max(1, int(len(all_speakers) * args.val_fraction))
    val_speakers = set(spk.name for spk in all_speakers[:n_val])
    print(f"Total TRAIN speakers: {len(all_speakers)}, val: {n_val}, train: {len(all_speakers)-n_val}")

    # ── PHONEME TASK ────────────────────────────────────────────────────────────
    if do_phoneme:
        train_rows, val_rows = [], []
        for dr_dir in sorted(train_dir.iterdir()):
            if not dr_dir.is_dir():
                continue
            if phoneme_allowed_drs and dr_dir.name.upper() not in phoneme_allowed_drs:
                continue
            for spk_dir in sorted(dr_dir.iterdir()):
                if not spk_dir.is_dir():
                    continue
                for phn_file in sorted(spk_dir.glob("*.PHN")):
                    wav_file = phn_file.with_suffix(".WAV")
                    if not wav_file.exists():
                        continue
                    rel_path = make_rel(wav_file)
                    for start, end, phoneme in parse_phn(phn_file):
                        row = {"file": rel_path, "start": start, "end": end, "phoneme": phoneme}
                        if spk_dir.name in val_speakers:
                            val_rows.append(row)
                        else:
                            train_rows.append(row)

        test_rows = []
        for dr_dir in sorted(test_dir.iterdir()):
            if not dr_dir.is_dir():
                continue
            if phoneme_allowed_drs and dr_dir.name.upper() not in phoneme_allowed_drs:
                continue
            for spk_dir in sorted(dr_dir.iterdir()):
                if not spk_dir.is_dir():
                    continue
                for phn_file in sorted(spk_dir.glob("*.PHN")):
                    if phn_file.stem.upper().startswith("SA"):
                        continue
                    wav_file = phn_file.with_suffix(".WAV")
                    if not wav_file.exists():
                        continue
                    rel_path = make_rel(wav_file)
                    for start, end, phoneme in parse_phn(phn_file):
                        test_rows.append({"file": rel_path, "start": start, "end": end, "phoneme": phoneme})

        print("\n-- Phoneme task --")
        for name, rows in [("timit_train", train_rows), ("timit_val", val_rows), ("timit_test", test_rows)]:
            df = pd.DataFrame(rows, columns=["file", "start", "end", "phoneme"])
            out_path = out_dir / f"{name}.csv"
            df.to_csv(out_path, index=False)
            print(f"  {name}: {len(df)} segments, {df['phoneme'].nunique()} phonemes")

    # ── DIALECT TASK ────────────────────────────────────────────────────────────
    if do_dialect:
        d_train_rows, d_val_rows = [], []
        for dr_dir in sorted(train_dir.iterdir()):
            if not dr_dir.is_dir():
                continue
            dialect = dr_dir.name.upper()
            for spk_dir in sorted(dr_dir.iterdir()):
                if not spk_dir.is_dir():
                    continue
                for wav_file in sorted(spk_dir.glob("*.WAV")):
                    rel_path = make_rel(wav_file)
                    row = {"file": rel_path, "dialect": dialect, "speaker": spk_dir.name}
                    if spk_dir.name in val_speakers:
                        d_val_rows.append(row)
                    else:
                        d_train_rows.append(row)

        d_test_rows = []
        for dr_dir in sorted(test_dir.iterdir()):
            if not dr_dir.is_dir():
                continue
            dialect = dr_dir.name.upper()
            for spk_dir in sorted(dr_dir.iterdir()):
                if not spk_dir.is_dir():
                    continue
                for wav_file in sorted(spk_dir.glob("*.WAV")):
                    rel_path = make_rel(wav_file)
                    d_test_rows.append({
                        "file": rel_path,
                        "dialect": dialect,
                        "speaker": spk_dir.name,
                    })

        print("\n-- Dialect task --")
        print(f"  Dialect regions: {DIALECT_LABELS}")
        for name, rows in [
            ("timit_dialect_train", d_train_rows),
            ("timit_dialect_val", d_val_rows),
            ("timit_dialect_test", d_test_rows),
        ]:
            df = pd.DataFrame(rows, columns=["file", "dialect", "speaker"])
            out_path = out_dir / f"{name}.csv"
            df.to_csv(out_path, index=False)
            counts = df["dialect"].value_counts().sort_index()
            print(f"  {name}: {len(df)} utterances | {dict(counts)}")

    print("\nDone.")
    if do_phoneme:
        print("  Phoneme: python3 -m nkululeko.nkululeko --config data/timit/exp.ini")
    if do_dialect:
        print("  Dialect: python3 -m nkululeko.nkululeko --config data/timit/exp_dialect.ini")


if __name__ == "__main__":
    main()
