# HAD - Half-Truth Audio Detection Dataset

## Overview

The **Half-Truth Audio Detection (HAD)** dataset is designed for detecting partially fake audio, where small fabricated audio clips are covertly inserted into authentic speech recordings. Unlike previous fake audio detection datasets that focus on entirely synthetic utterances, HAD addresses a more challenging and realistic scenario: identifying manipulated regions within otherwise genuine speech.

The partially manipulated audio samples in HAD involve minimal alterations, typically limited to modifying just a few words within an utterance. These altered segments are created using state-of-the-art speech synthesis technology.

## Dataset Information

- **Language**: Chinese
- **Size**: 8.1 GB
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Access**: Public
- **Target**: Partial deepfake detection
- **DOI**: [10.48550/arXiv.2104.03617](https://doi.org/10.48550/arXiv.2104.03617)

## Download

The dataset can be downloaded from Zenodo:
- **URL**: https://zenodo.org/records/10377492
- **File**: HAD.zip (8.1 GB, MD5: 4daef62a7cf20c71b052635c968ece1c)

## Citation

When using this dataset, please cite:

```bibtex
@inproceedings{yi2021halftruth,
  title={Half-Truth: A Partially Fake Audio Detection Dataset},
  author={Yi, Jiangyan and Bai, Ye and Tao, Jianhua and Ma, Haoxin and Tian, Zhengkun and Wang, Chenglong and Wang, Tao and Fu, Ruibo},
  booktitle={Interspeech},
  pages={1654--1658},
  year={2021}
}
```

## Dataset Features

- **Capability**: The dataset enables both:
  1. Identification of counterfeit utterances
  2. Pinpointing of manipulated regions within speech recordings

- **Manipulation Type**: Partial audio falsification with minimal alterations (typically a few words per utterance)

- **Synthesis Technology**: Created using state-of-the-art speech synthesis methods

## Usage with Nkululeko

After downloading and extracting the dataset:

1. Place the extracted HAD dataset in this directory (`data/had/`), or create a symbolic link:
   ```bash
   ln -sf /path/to/extracted/HAD /path/to/nkululeko/data/had/
   # extract the zipped file inside each partition/split, e.g., HAD_train/
   unzip train.zip
   ```

2. Process the database according to HAD's structure

3. Run experiments using Nkululeko configuration files (INI format)

## Authors

- Yi, Jiangyan
- Bai, Ye
- Tao, Jianhua
- Ma, Haoxin
- Tian, Zhengkun
- Wang, Chenglong
- Wang, Tao
- Fu, Ruibo

## Publication

Presented at **INTERSPEECH 2021**

## Keywords

- Fake Audio Detection
- Deepfake
- Partial Audio Manipulation
- Speech Synthesis
- Audio Forensics

