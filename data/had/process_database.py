#!/usr/bin/env python3
"""
Process the HAD (Half-Truth Audio Detection) dataset and generate CSV files.

The HAD dataset contains partially fake audio where small fabricated audio clips
are inserted into authentic speech recordings. The dataset is designed for
detecting partial audio falsification in Chinese speech.

Expected dataset structure (after extraction from HAD.zip):
HAD/
├── HAD_train/conbine
├── HAD_dev/conbine
└── HAD_test/test  


Output CSV files:
- had_train.csv
- had_dev.csv
- had_test.csv
- had.csv (all combined)

Columns: file, label
- file: path to the audio file
- label: 'real' or 'fake'
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Add nkululeko parent directory to path
script_dir = Path(__file__).parent
nkululeko_root = script_dir.parent.parent
sys.path.insert(0, str(nkululeko_root))

try:
    from nkululeko.utils.files import find_files
except ImportError:
    # Fallback to glob if nkululeko is not available
    def find_files(directory, ext=None, relative=False):
        """Find all files with given extensions in directory."""
        directory = Path(directory)
        if ext is None:
            ext = ["*"]
        files = []
        for extension in ext:
            files.extend(directory.rglob(f"*.{extension}"))
        return sorted([str(f) for f in files])


def read_label_file(label_file_path):
    """
    Read HAD label file and return a dictionary mapping filename to label.
    Format: filename timestamp-info overall_label
    overall_label: 0 = fake (has replacement), 1 = real (no replacement)
    """
    labels = {}
    if not label_file_path.exists():
        return labels
    
    with open(label_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                filename = parts[0]  # e.g., HAD_train_fake_00000001
                overall_label = parts[-1]  # Last column: 0 or 1
                # 0 = fake (has segment replacement)
                # 1 = real (no segment replacement)
                labels[filename] = 'real' if overall_label == '1' else 'fake'
    
    return labels


def process_split_directory_structure(data_dir, output_dir):
    """
    Process HAD dataset with HAD_train/HAD_dev/HAD_test directory structure.
    Files are in 'conbine' or 'test' subdirectories.
    Labels are read from label files (*_label.txt).
    """
    data_dir = Path(data_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define splits with HAD naming convention
    split_configs = {
        "train": {"dir": "HAD_train", "subdir": ["conbine"], "label_file": "HAD_train_label.txt"},
        "dev": {"dir": "HAD_dev", "subdir": ["conbine"], "label_file": "HAD_dev_label.txt"},
        "test": {"dir": "HAD_test", "subdir": ["test", "conbine"], "label_file": "HAD_test_label.txt"}
    }
    
    all_data = []
    
    for split_key, config in split_configs.items():
        split_dir = data_dir / config["dir"]
        
        if not split_dir.exists():
            print(f"WARNING: Split directory not found: {split_dir}")
            continue
        
        print(f"\nProcessing {split_key} split from {split_dir}")
        
        # Read label file
        label_file = split_dir / config["label_file"]
        labels_dict = read_label_file(label_file)
        
        if not labels_dict:
            print(f"  WARNING: No labels found in {label_file}")
        else:
            print(f"  Loaded {len(labels_dict)} labels from {label_file}")
        
        # Find the audio subdirectory
        audio_dir = None
        for subdir_name in config["subdir"]:
            candidate = split_dir / subdir_name
            if candidate.exists():
                audio_dir = candidate
                break
        
        if audio_dir is None:
            print(f"  WARNING: Audio subdirectory not found in {split_dir}")
            print(f"  Tried: {config['subdir']}")
            continue
        
        print(f"  Reading files from {audio_dir}")
        
        # Find all wav files
        wav_files = find_files(audio_dir, ext=["wav"], relative=False)
        
        if not wav_files:
            print(f"  WARNING: No .wav files found in {audio_dir}")
            continue
        
        data_list = []
        real_count = 0
        fake_count = 0
        unlabeled_count = 0
        
        # Process each file
        for f in wav_files:
            filename = Path(f).stem
            
            # First try to get label from label file
            if filename in labels_dict:
                label = labels_dict[filename]
            elif '_fake_' in filename:
                # Fallback: check filename pattern
                label = 'fake'
            elif '_real_' in filename:
                label = 'real'
            else:
                # Cannot determine label
                unlabeled_count += 1
                continue
            
            if label == 'real':
                real_count += 1
            else:
                fake_count += 1
            
            data_list.append({
                "file": str(Path(f).relative_to(data_dir.parent)),
                "label": label,
            })
        
        print(f"  Found {len(wav_files)} total audio files")
        print(f"  - Real: {real_count}")
        print(f"  - Fake: {fake_count}")
        if unlabeled_count > 0:
            print(f"  - Unlabeled (skipped): {unlabeled_count}")
        
        if not data_list:
            print(f"  WARNING: No valid audio files processed for {split_key}")
            continue
        
        # Create dataframe for this split
        split_df = pd.DataFrame(data_list)
        
        # Save split CSV
        csv_path = output_dir / f"had_{split_key}.csv"
        split_df.to_csv(csv_path, index=False)
        
        print(f"✓ Saved {len(split_df)} samples to {csv_path}")
        
        # Add to all_data for combined CSV
        all_data.append(split_df)
    
    # Create combined CSV with all splits
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_csv = output_dir / "had.csv"
        combined_df.to_csv(combined_csv, index=False)
        print(f"\n✓ Saved {len(combined_df)} total samples to {combined_csv}")
        print(f"  - Real: {len(combined_df[combined_df['label'] == 'real'])}")
        print(f"  - Fake: {len(combined_df[combined_df['label'] == 'fake'])}")
    else:
        print("\nERROR: No data processed. Check dataset structure.")
        return False
    
    print("\nDONE")
    return True


def process_metadata_file(data_dir, output_dir, metadata_file):
    """
    Process HAD dataset using a metadata CSV file.
    Assumes metadata has columns: filename, label
    """
    data_dir = Path(data_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metadata_path = data_dir / metadata_file
    if not metadata_path.exists():
        print(f"ERROR: Metadata file not found: {metadata_path}")
        return False
    
    print(f"Reading metadata from {metadata_path}")
    df = pd.read_csv(metadata_path)
    
    print(f"Metadata columns: {df.columns.tolist()}")
    print(f"Total samples in metadata: {len(df)}")
    
    # TODO: Adjust column mapping based on actual metadata structure
    # This is a template that needs to be customized
    
    print("\nPlease customize the metadata processing based on your file structure.")
    print("Current implementation supports directory structure only.")
    return False


def main(data_dir, output_dir, metadata_file=None):
    """
    Main processing function.
    
    Args:
        data_dir: Path to the HAD directory (extracted from HAD.zip)
        output_dir: Path to save the CSV files
        metadata_file: Optional metadata CSV filename if dataset uses metadata
    """
    data_dir = Path(data_dir).resolve()
    
    if not data_dir.exists():
        print(f"ERROR: Directory not found: {data_dir}")
        print("\nPlease extract HAD.zip first and ensure the path is correct.")
        return
    
    print(f"Processing HAD dataset from: {data_dir}")
    print(f"Output directory: {output_dir}")
    
    # Try directory structure approach first
    if metadata_file:
        success = process_metadata_file(data_dir, output_dir, metadata_file)
    else:
        success = process_split_directory_structure(data_dir, output_dir)
    
    if not success:
        print("\n" + "="*60)
        print("NOTE: You may need to adjust this script based on the")
        print("actual structure of the HAD dataset after extraction.")
        print("Please inspect the extracted files and modify accordingly.")
        print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process HAD (Half-Truth Audio Detection) dataset"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./HAD/",
        help="Path to the extracted HAD directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./",
        help="Path to the output directory for CSV files"
    )
    parser.add_argument(
        "--metadata_file",
        type=str,
        default=None,
        help="Optional metadata CSV filename if dataset uses metadata file"
    )
    args = parser.parse_args()
    
    main(args.data_dir, args.output_dir, args.metadata_file)
