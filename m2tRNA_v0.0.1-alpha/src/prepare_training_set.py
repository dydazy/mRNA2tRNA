import os
import pandas as pd


def merge_csv_files(input_dir, output_csv_path):
    csv_files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]
    dfs = []
    for csv_file in csv_files:
        chunk_df = pd.read_csv(os.path.join(input_dir, csv_file))
        dfs.append(chunk_df)
    concatenated_df = pd.concat(dfs, axis=1)
    concatenated_df.to_csv(output_csv_path, index=False)
    print(f"save merged CSV in {output_csv_path}")

