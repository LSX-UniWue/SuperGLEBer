#!/usr/bin/env python3
"""
Script to split train.csv files in HarmfulContent subdirectories into train and dev sets (80/20 split).
"""

import argparse
import os
from pathlib import Path

import pandas as pd


def split_train_data(base_dir: str, train_ratio: float = 0.8, random_state: int = 42):
    """
    Split train.csv files in all subdirectories into train and dev sets.

    Args:
        base_dir: Base directory containing subdirectories with train.csv files
        train_ratio: Ratio of data to keep in training set (default: 0.8)
        random_state: Random seed for reproducible splits (default: 42)
    """
    base_path = Path(base_dir)

    if not base_path.exists():
        print(f"Error: Directory {base_dir} does not exist")
        return

    # Find all subdirectories containing train.csv
    subdirs_with_train = []
    subdirs_with_test = []
    for subdir in base_path.iterdir():
        if subdir.is_dir():
            train_file = subdir / "train.csv"
            test_file = subdir / "test.csv"
            if train_file.exists():
                subdirs_with_train.append(subdir)
            if test_file.exists():
                subdirs_with_test.append(subdir)

    if not subdirs_with_train and not subdirs_with_test:
        print(f"No subdirectories with train.csv or test.csv found in {base_dir}")
        return

    print(f"Found {len(subdirs_with_train)} subdirectories with train.csv files:")
    for subdir in subdirs_with_train:
        print(f"  - {subdir.name}")

    print(f"Found {len(subdirs_with_test)} subdirectories with test.csv files:")
    for subdir in subdirs_with_test:
        print(f"  - {subdir.name}")

    # Process each subdirectory
    for subdir in subdirs_with_train:
        train_file = subdir / "train.csv"
        print(f"\nProcessing {subdir.name}/train.csv...")

        try:
            # Read the train.csv file
            df = pd.read_csv(train_file, sep=";", encoding="utf-8")
            print(f"  Original data shape: {df.shape}")

            # Shuffle the data
            df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

            # Calculate split point
            split_idx = int(len(df_shuffled) * train_ratio)

            # Split into train and dev
            train_split = df_shuffled[:split_idx]
            dev_split = df_shuffled[split_idx:]

            print(f"  Train split shape: {train_split.shape}")
            print(f"  Dev split shape: {dev_split.shape}")

            # Save the splits
            train_output = subdir / "train.csv"
            dev_output = subdir / "dev.csv"

            # Backup original file if it doesn't already exist
            backup_file = subdir / "train_original.csv"
            if not backup_file.exists():
                train_file.rename(backup_file)
                print(f"  Backed up original file to train_original.csv")

            # Write the new files
            train_split.to_csv(train_output, sep=";", index=False, encoding="utf-8")
            dev_split.to_csv(dev_output, sep=";", index=False, encoding="utf-8")

            print(f"  Created: {train_output}")
            print(f"  Created: {dev_output}")

            # Verify class distribution if applicable
            if "C2A" in df.columns:
                orig_dist = df["C2A"].value_counts(normalize=True)
                train_dist = train_split["C2A"].value_counts(normalize=True)
                dev_dist = dev_split["C2A"].value_counts(normalize=True)

                print(f"  Class distribution:")
                print(f"    Original: {orig_dist.to_dict()}")
                print(f"    Train:    {train_dist.to_dict()}")
                print(f"    Dev:      {dev_dist.to_dict()}")

        except Exception as e:
            print(f"  Error processing {subdir.name}: {str(e)}")
            continue

    # Process test.csv files to add empty "label" column
    for subdir in subdirs_with_test:
        test_file = subdir / "test.csv"
        print(f"\nProcessing {subdir.name}/test.csv...")

        try:
            # Read the test.csv file
            df_test = pd.read_csv(test_file, sep=";", encoding="utf-8")
            print(f"  Original test data shape: {df_test.shape}")

            # Check if "label" column already exists
            if "label" not in df_test.columns:
                # Add empty "label" column
                df_test["label"] = ""
                print(f"  Added empty 'label' column")

                # Save the updated test file
                df_test.to_csv(test_file, sep=";", index=False, encoding="utf-8")
                print(f"  Updated: {test_file}")
            else:
                print(f"  'label' column already exists, skipping")

        except Exception as e:
            print(f"  Error processing test file in {subdir.name}: {str(e)}")
            continue

    if subdirs_with_train:
        print(
            f"\nSplit complete! Files have been split with {train_ratio:.0%} for training and {1 - train_ratio:.0%} for development."
        )

    if subdirs_with_test:
        print(f"\nTest files processed! Added empty 'label' column where needed.")


def main():
    parser = argparse.ArgumentParser(description="Split train.csv files into train and dev sets")
    parser.add_argument(
        "--base_dir",
        default="data/Germeval/2025/HarmfulContent",
        help="Base directory containing subdirectories with train.csv files",
    )
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Ratio of data for training set (default: 0.8)")
    parser.add_argument(
        "--random_state", type=int, default=42, help="Random seed for reproducible splits (default: 42)"
    )

    args = parser.parse_args()

    # Convert relative path to absolute path if needed
    if not os.path.isabs(args.base_dir):
        script_dir = Path(__file__).parent
        base_dir = script_dir / args.base_dir
    else:
        base_dir = Path(args.base_dir)

    split_train_data(str(base_dir), args.train_ratio, args.random_state)


if __name__ == "__main__":
    main()
