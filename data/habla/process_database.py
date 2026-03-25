#!/usr/bin/env python3
"""
Process the HABLA (Hispanic American Balanced Latin-American) anti-spoofing dataset.

The HABLA dataset contains real (bonafide) and spoofed speech samples from Latin-American speakers.
It uses a protocol.txt file to define file metadata and labels.

Dataset structure (from RAR archive):
Latin_America_Spanish_anti_spoofing_dataset/
├── StarGAN/                    # Voice conversion (16,000 samples)
│   └── [accent-pair folders]/  # e.g., Argentina-Venezuela/
│       └── [speaker-pair]/     # e.g., arf_00295-vem_04310/
├── CycleGAN/                   # Voice conversion (16,000 samples)
│   └── [accent-pair folders]/
├── Diffusion/                  # Voice conversion (16,000 samples)
│   └── [accent-pair folders]/
├── TTS/                        # Text-to-speech (5,000 samples)
│   └── [accent folders]/       # e.g., Argentina/, Chile/
├── TTS-StarGAN/                # TTS + Voice conversion (2,500 samples)
│   └── [VC-type folders]/
├── TTS-Diff/                   # TTS + Voice conversion (2,500 samples)
│   └── [VC-type folders]/
├── protocol.txt                # Labels and metadata for all files
└── tree.txt                    # Directory structure reference

Protocol.txt format:
    Subject_id file_name - spoof_type Label
    
Examples:
    arf_00295 StarGAN-arf_00295_01349969200-cof_03349_0077577 - StarGAN spoof
    com_00001 com_00001_0001234 - - bonafide

Output CSV files:
- habla_train.csv
- habla_dev.csv  
- habla_test.csv
- habla.csv (all combined)

Columns: file, label, speaker
"""

import argparse
import os
import re
import sys
from pathlib import Path

import pandas as pd


# Spoof method folders in the dataset
SPOOF_FOLDERS = ['StarGAN', 'CycleGAN', 'Diffusion', 'TTS', 'TTS-StarGAN', 'TTS-Diff']


def parse_protocol(protocol_path, data_dir, file_map):
    """
    Parse the protocol.txt file to extract file metadata.
    
    Protocol format: Subject_id file_name - spoof_type Label
    
    For bonafide: com_00001 com_00001_0001234 - - bonafide
    For spoof: arf_00295 StarGAN-arf_00295_01349969200-cof_03349_0077577 - StarGAN spoof
    
    Args:
        protocol_path: Path to protocol.txt file
        data_dir: Path to the dataset root directory
        file_map: Dictionary mapping filenames to full paths
        
    Returns:
        DataFrame with file, label, speaker columns
    """
    data = []
    
    with open(protocol_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Parse the line - format varies but last word is always label
            parts = line.split()
            if len(parts) < 3:
                continue
            
            speaker_id = parts[0]
            raw_label = parts[-1].lower()  # 'spoof' or 'bonafide'
            
            # Extract filename (second element, may contain hyphens)
            # Find filename by looking for the pattern before " - "
            match = re.match(r'^(\S+)\s+(\S+)\s+-\s+(\S+)\s+(\S+)$', line)
            _label_map = {"bonafide": "real", "spoof": "fake"}
            if match:
                speaker_id = match.group(1)
                filename = match.group(2)
                spoof_type = match.group(3)
                raw_label = match.group(4).lower()
            else:
                # Fallback: assume second part is filename
                filename = parts[1]
            label = _label_map.get(raw_label, raw_label)
            
            # Find the actual file path
            file_path = None
            
            # Try exact match first
            if filename in file_map:
                file_path = file_map[filename]
            # Try with .wav extension
            elif filename + '.wav' in file_map:
                file_path = file_map[filename + '.wav']
            # Try stem match
            else:
                for key, path in file_map.items():
                    if Path(key).stem == filename or key.startswith(filename):
                        file_path = path
                        break
            
            if file_path:
                rel_path = file_path.relative_to(data_dir.parent)
                data.append({
                    'file': str(rel_path),
                    'label': label,
                    'speaker': speaker_id
                })
    
    return pd.DataFrame(data)


def find_audio_files(data_dir, extensions=None):
    """
    Recursively find all audio files in the dataset directory.
    
    Args:
        data_dir: Path to search for audio files
        extensions: List of file extensions to search for
        
    Returns:
        Dictionary mapping filename (with and without extension) to full paths
    """
    if extensions is None:
        extensions = ['.wav', '.flac', '.mp3']
    
    data_dir = Path(data_dir)
    file_map = {}
    
    for ext in extensions:
        for audio_file in data_dir.rglob(f'*{ext}'):
            # Map by stem (without extension)
            file_map[audio_file.stem] = audio_file
            # Map by full filename
            file_map[audio_file.name] = audio_file
    
    return file_map


def process_from_directory(data_dir):
    """
    Process dataset by scanning directories if protocol.txt is not available.
    Determines label based on folder structure.
    
    Args:
        data_dir: Path to the dataset root directory
        
    Returns:
        DataFrame with file, label, speaker columns
    """
    data_dir = Path(data_dir)
    data = []
    
    for audio_file in data_dir.rglob('*.wav'):
        # Determine if spoof or bonafide based on parent folders
        rel_parts = audio_file.relative_to(data_dir).parts
        
        # Check if any parent folder is a spoof method
        is_spoof = any(folder in SPOOF_FOLDERS for folder in rel_parts)
        # Normalize labels to match other deepfake datasets: bonafide->real, spoof->fake
        label = 'fake' if is_spoof else 'real'
        
        # Extract speaker ID from filename
        # Format: accent_speaker (e.g., arf_00295)
        stem = audio_file.stem
        # For spoof files, format is like: StarGAN-arf_00295_xxx-cof_03349_xxx
        # Extract the source speaker
        if '-' in stem and any(stem.startswith(sf) for sf in SPOOF_FOLDERS):
            # Remove spoof method prefix and extract first speaker
            parts = stem.split('-')
            if len(parts) >= 2:
                speaker_part = parts[1]
                speaker_match = re.match(r'([a-z]{2,3}_\d+)', speaker_part)
                speaker = speaker_match.group(1) if speaker_match else speaker_part[:10]
            else:
                speaker = stem[:10]
        else:
            # Bonafide file: format is accent_speaker_id
            speaker_match = re.match(r'([a-z]{2,3}_\d+)', stem)
            speaker = speaker_match.group(1) if speaker_match else stem.split('_')[0]
        
        rel_path = audio_file.relative_to(data_dir.parent)
        
        data.append({
            'file': str(rel_path),
            'label': label,
            'speaker': speaker
        })
    
    return pd.DataFrame(data)


def process_database(data_dir, output_dir):
    """
    Process the HABLA dataset and generate CSV files.
    
    Args:
        data_dir: Path to the dataset directory (Latin_America_Spanish_anti_spoofing_dataset)
        output_dir: Path to save the CSV files
    """
    data_dir = Path(data_dir).resolve()
    output_dir = Path(output_dir).resolve()
    
    if not data_dir.is_dir():
        raise FileNotFoundError(f"ERROR: Directory not found: {data_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # First, find all audio files
    print("Scanning for audio files...")
    file_map = find_audio_files(data_dir)
    print(f"Found {len(file_map) // 2} audio files")  # Divided by 2 since we map both stem and name
    
    # Try to use protocol.txt if available
    protocol_file = data_dir / 'protocol.txt'
    
    if protocol_file.exists():
        print(f"Found protocol.txt, parsing metadata...")
        df = parse_protocol(protocol_file, data_dir, file_map)
        
        if len(df) == 0:
            print("WARNING: Could not match files from protocol.txt, falling back to directory scan")
            df = process_from_directory(data_dir)
    else:
        print("No protocol.txt found, scanning directories...")
        df = process_from_directory(data_dir)
    
    if len(df) == 0:
        raise ValueError("No audio files found in the dataset directory")
    
    print(f"\nFound {len(df)} total samples")
    print(f"  - Bonafide: {len(df[df['label'] == 'real'])}")
    print(f"  - Spoof: {len(df[df['label'] == 'fake'])}")
    
    # Shuffle and split (60% train, 20% dev, 20% test)
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    total = len(df_shuffled)
    
    train_end = int(0.6 * total)
    dev_end = int(0.8 * total)
    
    df_train = df_shuffled.iloc[:train_end]
    df_dev = df_shuffled.iloc[train_end:dev_end]
    df_test = df_shuffled.iloc[dev_end:]
    
    # Save CSV files
    train_file = output_dir / 'habla_train.csv'
    dev_file = output_dir / 'habla_dev.csv'
    test_file = output_dir / 'habla_test.csv'
    all_file = output_dir / 'habla.csv'
    
    df_train.to_csv(train_file, index=False)
    df_dev.to_csv(dev_file, index=False)
    df_test.to_csv(test_file, index=False)
    df_shuffled.to_csv(all_file, index=False)
    
    print(f"\n✓ Saved {len(df_train)} samples to {train_file}")
    print(f"✓ Saved {len(df_dev)} samples to {dev_file}")
    print(f"✓ Saved {len(df_test)} samples to {test_file}")
    print(f"✓ Saved {len(df_shuffled)} samples to {all_file}")
    
    print("\nLabel distribution:")
    print(f"  Train - Bonafide: {len(df_train[df_train['label'] == 'real'])}, Spoof: {len(df_train[df_train['label'] == 'fake'])}")
    print(f"  Dev   - Bonafide: {len(df_dev[df_dev['label'] == 'real'])}, Spoof: {len(df_dev[df_dev['label'] == 'fake'])}")
    print(f"  Test  - Bonafide: {len(df_test[df_test['label'] == 'real'])}, Spoof: {len(df_test[df_test['label'] == 'fake'])}")
    
    # Print unique speakers
    print(f"\nUnique speakers: {df['speaker'].nunique()}")
    
    print("\nDONE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process HABLA anti-spoofing dataset and generate CSV files"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./HABLA/",
        help="Path to the HABLA dataset directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./",
        help="Path to the output directory for CSV files"
    )
    args = parser.parse_args()
    
    process_database(args.data_dir, args.output_dir)
