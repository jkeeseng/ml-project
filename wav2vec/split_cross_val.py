import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

def prepare_audio_dataset_for_cross_validation(csv_path, output_base_path, n_splits=3):
    """
    Prepares an audio dataset for cross-validation by splitting it into
    multiple folds and saving each fold's train and validation data to CSV files.

    Args:
    - csv_path (str): The path to the CSV file containing the dataset.
    - output_base_path (str): The base path where the train and validation set CSVs for each fold will be saved.
    - n_splits (int): The number of folds for cross-validation (default is 3).
    """
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_path)

    # Initialize the KFold object
    kf = KFold(n_splits=n_splits, shuffle=True)

    # Loop through each fold
    fold = 0
    for train_index, val_index in kf.split(df):
        fold += 1
        train_df, val_df = df.iloc[train_index], df.iloc[val_index]

        # Construct file paths for this fold's train and validation CSVs
        train_csv_path = f'{output_base_path}/train_files_fold_{fold}.csv'
        val_csv_path = f'{output_base_path}/val_files_fold_{fold}.csv'

        # Save the train and validation DataFrames to CSV files
        train_df.to_csv(train_csv_path, index=False)
        val_df.to_csv(val_csv_path, index=False)

        print(f"Fold {fold}:")
        print(f"Training dataset saved to {train_csv_path}")
        print(f"Validation dataset saved to {val_csv_path}\n")


if __name__ == "__main__":
    # Example usage:
    csv_path = 'data/filtered_dataset_sepedi/processed_files.csv'
    output_base_path = 'data/filtered_dataset_sepedi/'

    prepare_audio_dataset_for_cross_validation(csv_path, output_base_path, n_splits=3)
