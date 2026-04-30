# Pre-processing script for DementiaBank

This dataset needs registration to access. You can find the registration link [here](https://dementia.talkbank.org/).

## Structure of the dataset (DEMENTIABANK directory)
The dataset is organized into a directory structure that contains audio files categorized by language, speaker type, and session. Below is an example of the directory structure:

```bash
DEMENTIABANK/
├── English/
│   ├── Pitt/
│   │   ├── Control/
│   │   │   ├── coockie/
│   │   │   │   ├── 002-0.mp3
│   │   │   │   ├── 002-1.mp3
│   │   │   │   └── ...
│   │   └── Dementia/
│   │       ├── dementia/
│   │       │   ├── 001-0.mp3
│   │       │   ├── 001-1.mp3
│   │       │   └── ...
```

## Pre-processing steps

```bash
python process_database.py
cd ../..
python3 -m nkululeko.nkululeko --config data/dementiabank/exp.ini
```

> [!IMPORTANT]
> Use of the Pitt corpus requires acknowledgment of this grant support: NIA AG03705 and AG05133. (Pitt corpus was supported by the National Institutes of Health grants NIA AG03705 and AG05133.)

Reference:  
[1] Becker, J. T., Boller, F., Lopez, O. L., Saxton, J., & McGonigle, K. L. (1994). The natural history of Alzheimer's disease: description of study cohort and accuracy of diagnosis. Archives of Neurology, 51(6), 585-594.  

## Citation:  
Please cite the following paper if you use this data processing script for your research:
Atmaja, B. T., & Sakti, S. (2025). Dementia Prediction From Speech Signal Using Optimized Prosodic Features. APSIPA Annual Summit and Conference.
```bibtex
@inproceedings{AtmajaDementia2025,
address = {Singapore},
author = {Atmaja, Bagus Tris and Sakti, Sakriani},
booktitle = {APSIPA Annu. Summit Conf.},
title = {{Dementia Prediction From Speech Signal Using Optimized Prosodic Features}},
year = {2025}
}
```