# Nkululeko pre-processing for ASVspoof 2019 dataset

## Dataset description

The ASVspoof 2019 Logical Access (LA) database is the 3rd edition of the ASVspoof challenge dataset. It contains bona fide and spoofed speech utterances for training countermeasure (CM) systems against text-to-speech (TTS) and voice conversion (VC) attacks.

- **Train set**: 25,380 samples (2,580 real, 22,800 fake)
- **Dev set**: 24,844 samples (2,548 real, 22,296 fake)
- **Eval set**: 71,237 samples (7,355 real, 63,882 fake)
- **Labels**: `real` (bonafide), `fake` (spoof)
- **Audio**: 16 kHz, 16-bit FLAC

### Spoofing Systems

| ID  | Type   | Method                     |
|-----|--------|----------------------------|
| A01 | TTS    | Neural waveform model      |
| A02 | TTS    | Vocoder                    |
| A03 | TTS    | Vocoder                    |
| A04 | TTS    | Waveform concatenation     |
| A05 | VC     | Vocoder                    |
| A06 | VC     | Spectral filtering         |
| A07 | TTS    | Vocoder + GAN              |
| A08 | TTS    | Neural waveform            |
| A09 | TTS    | Vocoder                    |
| A10 | TTS    | Neural waveform            |
| A11 | TTS    | Griffin-Lim                |
| A12 | TTS    | Neural waveform            |
| A13 | TTS+VC | Waveform concatenation + filtering |
| A14 | TTS+VC | Vocoder                    |
| A15 | TTS+VC | Neural waveform            |
| A16 | TTS    | Waveform concatenation     |
| A17 | VC     | Waveform filtering         |
| A18 | VC     | Vocoder                    |
| A19 | VC     | Spectral filtering         |

A01-A06 appear in train/dev; A07-A19 are unseen attacks in the eval set.

## Pre-processing command

Download from: [https://datashare.ed.ac.uk/handle/10283/3336](https://datashare.ed.ac.uk/handle/10283/3336)

```bash
# Extract the downloaded archive into the LA/ directory, then generate CSV files:
cd data/asvspoof-2019
python3 process_database.py --data_dir ./LA

# Run experiment
cd ../..
python3 -m nkululeko.nkululeko --config data/asvspoof-2019/exp.ini
```

## Dataset structure

```
asvspoof-2019/
├── LA/
│   ├── ASVspoof2019_LA_train/flac/    # LA_T_*.flac
│   ├── ASVspoof2019_LA_dev/flac/      # LA_D_*.flac
│   ├── ASVspoof2019_LA_eval/flac/     # LA_E_*.flac
│   ├── ASVspoof2019_LA_cm_protocols/
│   │   ├── ASVspoof2019.LA.cm.train.trn.txt
│   │   ├── ASVspoof2019.LA.cm.dev.trl.txt
│   │   └── ASVspoof2019.LA.cm.eval.trl.txt
│   └── README.LA.txt
├── asv2019.csv         # Combined (all splits, shuffled)
├── asv2019_train.csv
├── asv2019_dev.csv
├── asv2019_test.csv
├── process_database.py
└── README.md
```

## CSV format

```
file,speaker,label
./LA/ASVspoof2019_LA_train/flac/LA_T_1138215.flac,LA_0079,real
./LA/ASVspoof2019_LA_train/flac/LA_T_1223219.flac,LA_0085,fake
```

## References

[1] Todisco et al., "ASVspoof 2019: Future Horizons in Spoofed and Fake Audio Detection," Interspeech 2019.

[2] Wang et al., "ASVspoof 2019: A Large-Scale Public Database of Synthesized, Converted and Replayed Speech," Computer Speech & Language, 2020.

[3] Dataset: https://datashare.ed.ac.uk/handle/10283/3336
