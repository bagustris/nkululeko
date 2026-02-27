import os
import pandas as pd
import argparse

def main(data_dir):
    """Process ASVspoof 2021 Deepfake (DF) evaluation dataset."""

    key_file = os.path.join(data_dir, "DF-keys-full/keys/DF/CM/trial_metadata.txt")
    parts = ["part00", "part01", "part02"]
    
    print(f"Reading metadata from {key_file}...")
    try:

        df = pd.read_csv(key_file, sep=r'\s+', header=None, low_memory=False)
        df = df[[0, 1, 5]].copy()
        df.columns = ['speaker', 'file_id', 'raw_label']
    except FileNotFoundError:
        print(f"Error: Metadata file not found at {key_file}")
        return
    except pd.errors.ParserError as e:
        print(f"Error parsing metadata file {key_file}: {e}")
        return

    df["label"] = df["raw_label"].map({"bonafide": "real", "spoof": "fake"})

    print("Indexing audio files across parts (this may take a moment)...")
    file_to_path = {}
    for part in parts:
        part_dir = os.path.join(data_dir, f"ASVspoof2021_DF_eval_{part}/ASVspoof2021_DF_eval/flac")
        if os.path.exists(part_dir):
            for f in os.listdir(part_dir):
                if f.endswith(".flac"):

                    file_to_path[f.replace(".flac", "")] = os.path.join(part_dir, f)

    print("Mapping metadata to file paths...")
    df["file"] = df["file_id"].map(file_to_path)
    
    df_final = df.dropna(subset=['file', 'label'])[['file', 'speaker', 'label']]
    df_final = df_final.reset_index(drop=True)

    output_file = os.path.join(data_dir, "asv2021_df_eval.csv")
    df_final.to_csv(output_file, index=False)

    print(f"✓ Created {output_file} with {len(df_final)} samples")
    
    print("\nLabel distribution in DF evaluation set:")
    print(df_final["label"].value_counts())
    
    total_speakers = df_final["speaker"].nunique()
    print(f"Total speakers: {total_speakers}")

    missing_count = len(df) - len(df_final)
    if missing_count > 0:
        print(f"Warning: {missing_count} files listed in metadata were not found in {parts}")

    print("\nDone!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process ASVspoof 2021 DF Evaluation dataset"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Path to the ASVspoof 2021 DF root directory",
    )
    args = parser.parse_args()

    main(args.data_dir)