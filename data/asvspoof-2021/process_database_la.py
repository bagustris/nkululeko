import os
import pandas as pd
import argparse


def main(data_dir):
    """Process ASVspoof 2021 Logical Access (LA) evaluation dataset.

    Handles the single-folder layout:
        <data_dir>/ASVspoof2021_LA_eval/flac/            -- audio files
        <data_dir>/ASVspoof2021_LA_eval/trial_metadata.txt  -- ground-truth metadata (preferred)
        <data_dir>/ASVspoof2021_LA_eval/ASVspoof2021.LA.cm.eval.trl.txt  -- ID-only trial list (fallback)

    trial_metadata.txt format (space-separated, no header):
        speaker_id  file_id  codec  channel  system_id  key  trim  subset
    where key is 'bonafide' or 'spoof', mapped to 'real' / 'fake'.
    """

    eval_dir = os.path.join(data_dir, "ASVspoof2021_LA_eval")
    flac_dir = os.path.join(eval_dir, "flac")
    metadata_file = os.path.join(eval_dir, "trial_metadata.txt")
    trial_file = os.path.join(eval_dir, "ASVspoof2021.LA.cm.eval.trl.txt")

    label_map = {"bonafide": "real", "spoof": "fake"}

    if os.path.isfile(metadata_file):
        # --- Read ground-truth metadata ---
        print(f"Reading ground-truth metadata from {metadata_file}...")
        try:
            df = pd.read_csv(
                metadata_file,
                sep=" ",
                header=None,
                names=[
                    "speaker",
                    "file_id",
                    "codec",
                    "channel",
                    "system_id",
                    "key",
                    "trim",
                    "subset",
                ],
            )
        except Exception as e:
            print(f"Error parsing metadata file {metadata_file}: {e}")
            return
        df["label"] = df["key"].map(label_map)
        unmapped = df["label"].isna().sum()
        if unmapped:
            print(
                f"Warning: {unmapped} rows with unrecognised key values (not bonafide/spoof) — set to 'unknown'"
            )
            df["label"] = df["label"].fillna("unknown")
        label_counts = df["label"].value_counts().to_dict()
        print(f"Label distribution from metadata: {label_counts}")
    else:
        # --- Fallback: ID-only trial list, no ground-truth labels ---
        print(f"trial_metadata.txt not found; falling back to {trial_file}...")
        print("Warning: ground-truth labels unavailable — all labels set to 'unknown'.")
        try:
            df = pd.read_csv(trial_file, header=None, names=["file_id"])
        except FileNotFoundError:
            print(f"Error: Trial list not found at {trial_file}")
            return
        except pd.errors.ParserError as e:
            print(f"Error parsing trial list {trial_file}: {e}")
            return
        df["speaker"] = "unknown"
        df["label"] = "unknown"

    # --- Index audio files ---
    print(f"Indexing audio files in {flac_dir} (this may take a moment)...")
    if not os.path.exists(flac_dir):
        print(f"Error: flac directory not found at {flac_dir}")
        return

    file_to_path = {}
    for f in os.listdir(flac_dir):
        if f.endswith(".flac"):
            file_to_path[f.replace(".flac", "")] = os.path.join(flac_dir, f)

    print(f"Found {len(file_to_path)} audio files.")

    # --- Map file IDs to full paths ---
    print("Mapping trial list to file paths...")
    df["file"] = df["file_id"].map(file_to_path)

    df_final = df.dropna(subset=["file"])[["file", "speaker", "label"]]
    df_final = df_final.reset_index(drop=True)

    output_file = os.path.join(data_dir, "asv2021_la_eval.csv")
    df_final.to_csv(output_file, index=False)

    print(f"✓ Created {output_file} with {len(df_final)} samples")

    missing_count = len(df) - len(df_final)
    if missing_count > 0:
        print(
            f"Warning: {missing_count} file IDs in the trial list were not found in {flac_dir}"
        )

    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process ASVspoof 2021 LA Evaluation dataset"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Path to the root directory containing ASVspoof2021_LA_eval/",
    )
    args = parser.parse_args()

    main(args.data_dir)
