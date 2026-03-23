# Nkululeko pre-processing for PartialSpoof dataset

## Dataset description

The PartialSpoof database contains partially spoofed audio samples where synthesized or transformed speech segments are embedded into bona fide utterances. Unlike existing databases that contain fully spoofed utterances, this dataset addresses a more realistic attack scenario.

The dataset is built upon the ASVspoof 2019 LA database and includes:
- **Train set**: 25,380 samples
- **Dev set**: 24,844 samples  
- **Eval set**: 71,237 samples
- **Labels**: Utterance-level (spoof/bonafide) and segment-level labels at multiple temporal resolutions

### Spoofing Methods

The dataset uses TTS and VC methods from ASVspoof 2019 (A01-A19) plus PartialSpoof concatenation (CON):
- A01-A06: Training/Dev set (TTS and VC)
- A07-A19: Evaluation set (unseen attacks)
- CON: Partially spoofed audio with embedded fake segments

## Pre-processing command

Download link: [https://zenodo.org/record/5766198](https://zenodo.org/record/5766198)

```bash
# Download all tar.gz files from Zenodo into PartialSpoof directory
# Then extract each archive:
cd PartialSpoof
for i in ./*.tar.gz; do tar -xvzf $i; done

# Generate CSV files
cd ..
python3 process_database.py

# Run experiment with EER metric (default for deepfake detection)
cd ../..
python3 -m nkululeko.nkululeko --config data/partialspoofing/exp.ini
```

## Dataset Structure

```
PartialSpoof/
├── database/
│   ├── train/
│   │   ├── con_wav/          # Audio files (LA_*.wav, CON_*.wav)
│   │   └── train.lst
│   ├── dev/
│   │   ├── con_wav/
│   │   └── dev.lst
│   ├── eval/
│   │   ├── con_wav/
│   │   └── eval.lst
│   ├── protocols/
│   │   ├── PartialSpoof_LA_cm_protocols/  # CM protocol files
│   │   └── ...
│   ├── segment_labels/       # Segment-level labels (.npy)
│   └── vad/                  # VAD timestamp annotations
└── README_v1.2
```

## Protocol Format

Protocol files follow ASVspoof format:
```
SPEAKER_ID AUDIO_FILE_NAME - SYSTEM_ID KEY
LA_0079 CON_T_0000029 - CON spoof
LA_0079 LA_T_1234567 - - bonafide
```

## References

[1] Zhang et al., "An Initial Investigation for Detecting Partially Spoofed Audio," Interspeech 2021.

[2] Zhang et al., "The PartialSpoof Database and Countermeasures for the Detection of Short Fake Speech Segments Embedded in an Utterance," IEEE/ACM TASLP, 2023.

[3] Dataset: https://zenodo.org/record/5766198
