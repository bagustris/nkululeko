# Nkululeko pre-processing for HABLA dataset

## Dataset description

The HABLA (Hispanic American Balanced Latin-American) dataset is a Latin-American voice anti-spoofing dataset containing samples of spoof and real human speech with different accents from Latin-American countries.

The dataset contains:
- **Bonafide samples**: 22,816 real speech samples from 162 speakers across 5 accents
- **Spoof samples**: 58,000 synthetic speech samples generated using various methods
- **Total**: 80,816 audio samples
- **Sampling rate**: 16 kHz
- **Language**: Spanish (Latin-American variants)

### Accents and Distribution

| Accent | Male Speakers | Female Speakers | Male Files | Female Files |
|--------|---------------|-----------------|------------|--------------|
| Colombian | 17 | 14 | 2,534 | 2,070 |
| Chilean | 17 | 12 | 2,487 | 1,602 |
| Peruvian | 20 | 18 | 2,917 | 2,529 |
| Venezuelan | 12 | 10 | 1,754 | 1,463 |
| Argentinian | 12 | 30 | 1,670 | 3,790 |

### Spoofing Methods

| Method | Type | # Samples |
|--------|------|-----------|
| StarGAN | Voice conversion | 16,000 |
| CycleGAN | Voice conversion | 16,000 |
| Diffusion | Voice conversion | 16,000 |
| TTS | Text-to-speech | 5,000 |
| TTS-StarGAN | TTS + Voice conversion | 2,500 |
| TTS-Diff | TTS + Voice conversion | 2,500 |

## Pre-processing command

Download link: [https://zenodo.org/record/7370805](https://zenodo.org/record/7370805)

```bash
# Download and extract the dataset
wget https://zenodo.org/record/7370805/files/Latin_America_Spanish_anti_spoofing_dataset.rar
unrar x Latin_America_Spanish_anti_spoofing_dataset.rar

# Generate CSV files
python3 process_database.py

# Run experiment (use an example INI file as template, e.g. examples/exp_emodb_os_xgb.ini)
cd ../..
python3 -m nkululeko.nkululeko --config path/to/your_config.ini
```

## File Naming Convention

Files follow this nomenclature:
- **com/cof**: Colombian male/female
- **clm/clf**: Chilean male/female  
- **pem/pef**: Peruvian male/female
- **vem/vef**: Venezuelan male/female
- **arm/arf**: Argentinian male/female

## References

[1] Zenodo Dataset: https://zenodo.org/record/7370805

[2] StarGAN-VC: Kameoka et al., "StarGAN-VC: Non-parallel many-to-many Voice Conversion Using Star Generative Adversarial Networks," 2018.

[3] CycleGAN-VC: Kaneko and Kameoka, "CycleGAN-VC: Non-parallel voice conversion using cycle-consistent adversarial networks," IEEE SLT 2018.
