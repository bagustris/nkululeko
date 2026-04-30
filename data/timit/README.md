# TIMIT Phoneme Recognition

TIMIT Acoustic-Phonetic Continuous Speech Corpus for 39-class phoneme recognition using nkululeko.

## Dataset

TIMIT contains broadband recordings of 630 speakers (462 train, 168 test) reading 10 sentences each. The corpus provides word and phoneme-level transcriptions with time boundaries.

- **Access**: Restricted — license required from [LDC (LDC93S1)](https://catalog.ldc.upenn.edu/LDC93S1)
- **Sample rate**: 16 kHz, 16-bit
- **Splits**: TRAIN (4,620 utterances) / TEST (1,680 utterances)

Place or symlink the TIMIT directory at `data/timit/TIMIT/` so the structure is:
```
data/timit/TIMIT/
├── TRAIN/
│   ├── DR1/
│   │   ├── FCJF0/
│   │   │   ├── SA1.WAV
│   │   │   ├── SA1.PHN
│   │   │   └── ...
│   └── ...
└── TEST/
    └── ...
```

## Phoneme Set

The 61 TIMIT phonemes are mapped to the standard **39-class** set (Lee & Hon, 1989):

| Reduction | Example |
|-----------|---------|
| `ix`, `ih` → `ih` | |
| `ax`, `ax-h` → `ah` | |
| `axr` → `er` | |
| `el` → `l`, `em` → `m`, `en`/`nx` → `n` | sonorant allophones |
| `hv` → `hh` | |
| `ux` → `uw` | |
| `zh` → `sh` | |
| `eng` → `ng` | |
| closures, `h#`, `pau`, `epi`, `q` → excluded | silence/closure |

## Pre-processing

```bash
cd /path/to/nkululeko
# Quick demo (DR1 + DR2 only, ~30K segments, ~6 min feature extraction):
python data/timit/process_database.py --dialect-regions DR1 DR2

# Full dataset (all 8 dialect regions, ~125K segments, several hours):
python data/timit/process_database.py
```

Options:
```
--timit-dir        Path to TIMIT root (default: ./data/timit/TIMIT)
--out-dir          Output directory for CSV files (default: ./data/timit)
--val-fraction     Fraction of train speakers for validation (default: 0.1)
--seed             Random seed for speaker split (default: 42)
--dialect-regions  Limit to specific DRs, e.g. DR1 DR2 (default: all 8)
```

Quick demo output (DR1+DR2):
- `timit_train.csv` — ~30K phoneme segments from 103 speakers
- `timit_val.csv`   — ~3K phoneme segments from 11 speakers
- `timit_test.csv`  — ~9K phoneme segments (SA excluded)

Full dataset output (all regions):
- `timit_train.csv` — ~124K phoneme segments from 416 speakers
- `timit_val.csv`   — ~14K phoneme segments from 46 speakers
- `timit_test.csv`  — ~41K phoneme segments (SA excluded)

## Running the Experiments

### Phoneme recognition (39 classes, segmented)
```bash
python3 -m nkululeko.nkululeko --config data/timit/exp.ini
```
Uses openSMILE eGeMAPSv02 + XGBoost on ~30K phoneme segments (DR1+DR2 default).
Features are cached after the first run — subsequent runs are much faster.

### Dialect recognition (8 classes, utterance-level)
```bash
python3 -m nkululeko.nkululeko --config data/timit/exp_dialect.ini
```
Uses openSMILE eGeMAPSv02 + SVM on 4,620 whole utterances (all 8 DRs).
Feature extraction finishes in ~4 minutes.

**Expected dialect result**: UAR ≈ 0.14–0.18 with eGeMAPSv02 + SVM.
This is slightly above the 12.5% chance level for 8 classes, which is expected —
TIMIT dialect labels were designed for corpus stratification, not as a hard
classification benchmark. American English dialect differences are subtle at the
purely acoustic level. Better results require lexical/linguistic features or
larger dialect-diverse corpora.

## Reference

Garofolo, J. et al. (1993). *TIMIT Acoustic-Phonetic Continuous Speech Corpus*. LDC93S1. Philadelphia: LDC.
