# Drop all 'Ratings_*' columns from all .parquet files in a folder

import os
import pandas as pd

folder = '/Volumes/storage/stock-robot-data/processed_features_noinfo/'  # Change this to your folder path

for fname in os.listdir(folder):
    if fname.endswith('.parquet'):
        fpath = os.path.join(folder, fname)
        df = pd.read_parquet(fpath)
        ratings_cols = [col for col in df.columns if col.startswith('Rating_')]
        if ratings_cols:
            df = df.drop(columns=ratings_cols)
            df.to_parquet(fpath)
            print(f"Dropped {ratings_cols} from {fname}")
        else:
            print(f"No 'Ratings_' columns found in {fname}. Columns: {list(df.columns)}")