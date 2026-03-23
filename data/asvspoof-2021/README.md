# ASVspoof 2021 — Data Preparation

This directory contains scripts to prepare the **ASVspoof 2021** Logical Access (LA) and Deepfake (DF) evaluation datasets for use with [Nkululeko](https://github.com/felixbur/nkululeko).

## Directory Layout

```
asvspoof-2021/
├── ASVspoof2021_LA_eval/          # Raw LA evaluation data (download separately)
│   ├── flac/                      # Audio files
│   ├── trial_metadata.txt         # Ground-truth metadata (preferred)
│   └── ASVspoof2021.LA.cm.eval.trl.txt  # ID-only trial list (fallback)
├── ASVspoof2021_DF_eval/          # Raw DF evaluation data (download separately)
│   ├── flac/                      # Audio files
│   ├── trial_metadata.txt         # Ground-truth metadata (preferred)
│   └── ASVspoof2021.DF.cm.eval.trl.txt  # ID-only trial list (fallback)
├── process_database_la.py         # Processing script for LA track
├── process_database_df.py         # Processing script for DF track
├── asv2021_la_eval.csv            # Output: Nkululeko-ready index (LA)
└── asv2021_df_eval.csv            # Output: Nkululeko-ready index (DF)
```

## Labels

Both scripts map the original ASVspoof keys as follows:

| Original key | Nkululeko label |
|---|---|
| `bonafide` | `real` |
| `spoof` | `fake` |

If `trial_metadata.txt` is absent, the script falls back to the ID-only trial list and assigns `unknown` to all labels.

## Usage

### Logical Access (LA)

```bash
python process_database_la.py --data_dir /path/to/asvspoof-2021
```

Produces `asv2021_la_eval.csv` (~181 566 samples).

### Deepfake (DF)

```bash
python process_database_df.py --data_dir /path/to/asvspoof-2021
```

Produces `asv2021_df_eval.csv` (~152 955 samples).

`--data_dir` defaults to the directory containing the script, so both scripts can be run without arguments when executed from this folder.

## Output CSV Format

| Column | Description |
|---|---|
| `file` | Absolute path to the `.flac` audio file |
| `speaker` | Speaker ID from metadata (or `unknown`) |
| `label` | `real` / `fake` / `unknown` |

## Requirements

- Python 3.8+
- `pandas`

## Dataset Source

The raw audio and metadata must be obtained from the official ASVspoof 2021 challenge:
<https://www.asvspoof.org/index2021.html>
