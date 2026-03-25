#!/usr/bin/env python3
"""
Process the PartialSpoof dataset and generate CSV files for nkululeko.

The PartialSpoof dataset contains partially spoofed audio where synthesized
segments are embedded into bona fide utterances.

Dataset structure:
PartialSpoof/
├── database/
│   ├── train/
│   │   ├── con_wav/          # Audio files
│   │   └── train.lst
│   ├── dev/
│   │   ├── con_wav/
│   │   └── dev.lst
│   ├── eval/
│   │   ├── con_wav/
│   │   └── eval.lst
│   └── protocols/
│       └── PartialSpoof_LA_cm_protocols/
│           ├── PartialSpoof.LA.cm.train.trl.txt
│           ├── PartialSpoof.LA.cm.dev.trl.txt
│           └── PartialSpoof.LA.cm.eval.trl.txt

Protocol format:
    SPEAKER_ID AUDIO_FILE_NAME - SYSTEM_ID KEY
    LA_0079 CON_T_0000029 - CON spoof
    LA_0079 LA_T_1234567 - - bonafide

Output CSV files:
- partialspoofing_train.csv
- partialspoofing_dev.csv
- partialspoofing_test.csv
- partialspoofing.csv (all combined)

Columns: file, label, speaker
"""

import argparse
from pathlib import Path

import pandas as pd


def parse_protocol(protocol_path, wav_dir, data_dir_parent):
    """
    Parse a PartialSpoof protocol file.
    
    Protocol format: SPEAKER_ID AUDIO_FILE_NAME - SYSTEM_ID KEY
    
    Args:
        protocol_path: Path to the protocol .txt file
        wav_dir: Path to the con_wav directory containing audio files
        data_dir_parent: Parent directory for relative path calculation
        
    Returns:
        DataFrame with file, label, speaker columns
    """
    data = []
    
    with open(protocol_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) < 5:
                continue
            
            speaker_id = parts[0]  # LA_XXXX
            audio_id = parts[1]    # CON_T_XXXXXXX or LA_T_XXXXXXX
            # parts[2] is '-'
            # parts[3] is SYSTEM_ID (CON or - for bonafide)
            raw_label = parts[4].lower()  # 'spoof' or 'bonafide'
            _label_map = {"bonafide": "real", "spoof": "fake"}
            label = _label_map.get(raw_label, raw_label)

            # Construct audio file path
            audio_file = wav_dir / f"{audio_id}.wav"
            
            if audio_file.exists():
                rel_path = audio_file.relative_to(data_dir_parent)
                data.append({
                    'file': str(rel_path),
                    'label': label,
                    'speaker': speaker_id
                })
    
    return pd.DataFrame(data)


def process_database(data_dir, output_dir):
    """
    Process the PartialSpoof dataset and generate CSV files.
    
    Args:
        data_dir: Path to the PartialSpoof directory
        output_dir: Path to save the CSV files
    """
    data_dir = Path(data_dir).resolve()
    output_dir = Path(output_dir).resolve()
    
    if not data_dir.is_dir():
        raise FileNotFoundError(f"ERROR: Directory not found: {data_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Paths to protocol files and audio directories
    protocols_dir = data_dir / 'database' / 'protocols' / 'PartialSpoof_LA_cm_protocols'
    
    splits = {
        'train': {
            'protocol': protocols_dir / 'PartialSpoof.LA.cm.train.trl.txt',
            'wav_dir': data_dir / 'database' / 'train' / 'con_wav'
        },
        'dev': {
            'protocol': protocols_dir / 'PartialSpoof.LA.cm.dev.trl.txt',
            'wav_dir': data_dir / 'database' / 'dev' / 'con_wav'
        },
        'test': {
            'protocol': protocols_dir / 'PartialSpoof.LA.cm.eval.trl.txt',
            'wav_dir': data_dir / 'database' / 'eval' / 'con_wav'
        }
    }
    
    all_data = []
    
    for split_name, paths in splits.items():
        protocol_file = paths['protocol']
        wav_dir = paths['wav_dir']
        
        if not protocol_file.exists():
            print(f"WARNING: Protocol file not found: {protocol_file}")
            continue
        
        if not wav_dir.exists():
            print(f"WARNING: Audio directory not found: {wav_dir}")
            continue
        
        print(f"Processing {split_name} split...")
        df = parse_protocol(protocol_file, wav_dir, data_dir.parent)
        
        if len(df) == 0:
            print(f"WARNING: No samples found for {split_name}")
            continue
        
        # Save split CSV
        csv_path = output_dir / f'partialspoofing_{split_name}.csv'
        df.to_csv(csv_path, index=False)
        
        bonafide_count = len(df[df['label'] == 'real'])
        spoof_count = len(df[df['label'] == 'fake'])

        print(f"✓ Saved {len(df)} samples to {csv_path}")
        print(f"  - Bonafide: {bonafide_count}, Spoof: {spoof_count}")
        
        all_data.append(df)
    
    # Create combined CSV
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_csv = output_dir / 'partialspoofing.csv'
        combined_df.to_csv(combined_csv, index=False)
        
        print(f"\n✓ Saved {len(combined_df)} total samples to {combined_csv}")
        print(f"  - Bonafide: {len(combined_df[combined_df['label'] == 'real'])}")
        print(f"  - Spoof: {len(combined_df[combined_df['label'] == 'fake'])}")
        print(f"  - Unique speakers: {combined_df['speaker'].nunique()}")
    
    print("\nDONE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process PartialSpoof dataset and generate CSV files"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./PartialSpoof/",
        help="Path to the PartialSpoof directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./",
        help="Path to the output directory for CSV files"
    )
    args = parser.parse_args()
    
    process_database(args.data_dir, args.output_dir)
