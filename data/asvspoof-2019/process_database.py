import os
import pandas as pd
import argparse

def process_protocol(protocol_path, audio_dir):
    """Parses ASVspoof protocol files and returns a formatted DataFrame."""
    df = pd.read_csv(protocol_path, sep=' ', header=None, 
                     names=['speaker', 'file', 'sys_id', 'unused', 'label'])

    df['file'] = df['file'].apply(lambda x: os.path.join(audio_dir, 'flac', f"{x}.flac"))
    
    df['label'] = df['label'].map({'bonafide': 'real', 'spoof': 'fake'})
    
    return df[['file', 'speaker', 'label']]

def main(data_dir):
    """Process ASVspoof 2019 Logical Access (LA) dataset."""
    
    splits = {
        'train': ('ASVspoof2019_LA_train', 'ASVspoof2019.LA.cm.train.trn.txt'),
        'dev':   ('ASVspoof2019_LA_dev',   'ASVspoof2019.LA.cm.dev.trl.txt'),
        'test':  ('ASVspoof2019_LA_eval',  'ASVspoof2019.LA.cm.eval.trl.txt')
    }
    
    protocol_base = os.path.join(data_dir, 'ASVspoof2019_LA_cm_protocols')
    all_dfs = []

    for name, (folder, proto) in splits.items():
        proto_path = os.path.join(protocol_base, proto)
        audio_path = os.path.join(data_dir, folder)
        
        print(f"Reading {name} metadata from {proto_path}...")
        try:
            df_split = process_protocol(proto_path, audio_path)
        except FileNotFoundError:
            print(f"Warning: Protocol file not found at {proto_path}. Skipping '{name}' split.")
            continue

        output_file = os.path.join(data_dir, f"asv2019_{name}.csv")
        df_split.to_csv(output_file, index=False)
        print(f"✓ Created {output_file} with {len(df_split)} samples")

        print(f"Label distribution in {name} set:")
        print(df_split["label"].value_counts())
        print("-" * 30)
        
        all_dfs.append(df_split)

    df_all = pd.concat(all_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
    combined_file = os.path.join(data_dir, "asv2019.csv")
    df_all.to_csv(combined_file, index=False)
    
    print(f"✓ Created {combined_file} with {len(df_all)} samples (shuffled)")
    print("\nFinal Label distribution (Complete Dataset):")
    print(df_all["label"].value_counts())
    print(f"Total speakers: {df_all['speaker'].nunique()}")
    print("\nDone!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process ASVspoof 2019 LA dataset"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Path to the 'LA/LA' directory",
    )
    args = parser.parse_args()

    main(args.data_dir)