import os
import pandas as pd
import argparse

def main(data_dir):
    """Process ASVspoof 2021 Logical Access (LA) evaluation dataset."""

    key_file = os.path.join(data_dir, "LA-keys-full/keys/LA/CM/trial_metadata.txt")
    audio_dir = os.path.join(data_dir, "ASVspoof2021_LA_eval/ASVspoof2021_LA_eval/flac")

    print(f"Reading metadata from {key_file}...")

    try:

        df_raw = pd.read_csv(key_file, sep=r'\s+', header=None, low_memory=False)
    except FileNotFoundError:
        print(f"Error: Metadata file not found at {key_file}")
        return

    df = df_raw[[0, 1, 5]].copy()
    df.columns = ['speaker', 'file_id', 'raw_label']

    df["label"] = df["raw_label"].map({"bonafide": "real", "spoof": "fake"})

    df["file"] = df["file_id"].apply(lambda x: os.path.join(audio_dir, f"{x}.flac"))

    if not os.path.isdir(audio_dir):
        print(f"Warning: Audio directory not found at {audio_dir}")

    df_final = df[['file', 'speaker', 'label']].reset_index(drop=True)

    output_file = os.path.join(data_dir, "asv2021_la_eval.csv")
    df_final.to_csv(output_file, index=False)

    print(f"✓ Created {output_file} with {len(df_final)} samples")
    
    print("\nLabel distribution in evaluation set:")
    print(df_final["label"].value_counts())
    
    total_speakers = df_final["speaker"].nunique()
    print(f"Total speakers: {total_speakers}")
    print("\nDone!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process ASVspoof 2021 LA Evaluation dataset"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Path to the ASVspoof 2021 root directory (default: current script directory)",
    )
    args = parser.parse_args()

    main(args.data_dir)