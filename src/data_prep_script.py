import argparse
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def clean_data(input_path , output_path):    
    print(f"\n Loading raw data from {input_path}...")
    
    df = pd.read_csv(input_path)
    initial_shape = df.shape
    
    # strip the column names
    df.columns = df.columns.str.strip()
    
    columns_to_drop = [
        'Destination Port', 'Subflow Fwd Packets', 'Subflow Fwd Bytes',
        'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'CWE Flag Count',
        'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate', 
        'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate',
        'Fwd IAT Min', 'Bwd IAT Min', 'Flow IAT Min' 
    ]
    
    # drop the columns from the dataset
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')
    print(f"\n Dropped specified columns-> New shape: {df.shape}")
    
    print("-" * 40)

    print(f"\n Fixing Infinity and NaN values...")
    
    # replace inf with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna() # drop all the NaN values
    print(f"\n Dropped {initial_shape[0] - len(df)} corrupted rows")
    
    print("-" * 40)
    
    
    print(f"\n Encoding Categorical columns...")
    TARGET_COL = "Label"
    if TARGET_COL in df.columns:
        print(f"\n Encoding target variable: {TARGET_COL}")
        df[TARGET_COL] = df[TARGET_COL].astype(str)
        le = LabelEncoder()
        df[TARGET_COL] = le.fit_transform(df[TARGET_COL])
        
        # save mapping for inference later
        mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        print(f"\n Label Mapping: {mapping}")
    
    print("-" * 40)
    
    print(f"\n Saving clean dataset to {output_path}...")
    # output folder existed or not
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # index=False doesn't add any rows number to the cols
    df.to_csv(output_path, index=False)
    print(f"\n Data Preparation is completed! Clean data saved")

    









if __name__ == "__main__":
    # Setup argparse so the script can be run flexibly from the terminal
    parser = argparse.ArgumentParser(description="Clean raw network dataset for deep learning modeling.")
    parser.add_argument('--input', type=str, default='data/raw/dataset.csv', help='Path to raw CSV')
    parser.add_argument('--output', type=str, default='data/processed/clean_data.csv', help='Path to save clean CSV')
    args = parser.parse_args()
    
    clean_data(args.input, args.output)