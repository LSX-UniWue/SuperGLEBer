#!/usr/bin/env python3
"""
Script to join two CSV files from FlauschErkennung data on comment_id
and create a merged dataset with all columns.
Automatically splits training data into train/dev (80/20) and processes test data.
"""

import argparse
import csv
import os
import random


def merge_flausch_data(training_data_dir, test_data_dir, output_dir):
    """
    Merge the two FlauschErkennung CSV files on comment_id.
    Splits training data into train/dev (80/20) and processes test data separately.

    Args:
        training_data_dir (str): Path to directory containing training CSV files
        test_data_dir (str): Path to directory containing test CSV files
        output_dir (str): Path to output directory for merged files
    """

    # Training file paths
    training_comments_file = os.path.join(training_data_dir, "comments.csv")
    training_task1_file = os.path.join(training_data_dir, "task1.csv")

    # Test file paths
    test_comments_file = os.path.join(test_data_dir, "comments.csv")

    # Check if all required files exist
    for file_path in [training_comments_file, training_task1_file]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required training file not found: {file_path}")

    if not os.path.exists(test_comments_file):
        raise FileNotFoundError(f"Required test file not found: {test_comments_file}")

    print("Processing training data...")
    training_data = process_training_data(training_comments_file, training_task1_file)

    print("Processing test data...")
    test_data = process_test_data(test_comments_file)

    print("Splitting training data into train/dev (80/20)...")
    train_data, dev_data = split_train_dev(training_data, train_ratio=0.8)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Write training split
    write_csv_data(train_data, os.path.join(output_dir, "train.csv"))
    print(f"Training data saved to: {os.path.join(output_dir, 'train.csv')} ({len(train_data)} samples)")

    # Write development split
    write_csv_data(dev_data, os.path.join(output_dir, "dev.csv"))
    print(f"Development data saved to: {os.path.join(output_dir, 'dev.csv')} ({len(dev_data)} samples)")

    # Write test data
    write_csv_data(test_data, os.path.join(output_dir, "test.csv"))
    print(f"Test data saved to: {os.path.join(output_dir, 'test.csv')} ({len(test_data)} samples)")

    print("Processing complete!")


def process_training_data(comments_file, task1_file):
    """
    Process training data by merging comments and task1 labels.

    Returns:
        list: List of dictionaries containing merged training data
    """
    # Data structures to hold the data
    comments_data = {}  # key: (document, comment_id), value: comment data
    task1_data = {}  # key: (document, comment_id), value: task1 data

    # Load comments.csv
    print("Loading training comments...")
    with open(comments_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["document"], row["comment_id"])
            comments_data[key] = {
                "document": row["document"],
                "comment_id": row["comment_id"],
                "comment": row["comment"],
            }

    # Load task1.csv
    print("Loading task1 data...")
    with open(task1_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["document"], row["comment_id"])
            task1_data[key] = {"flausch": row["flausch"]}

    print(f"Loaded {len(comments_data)} training comments")
    print(f"Loaded {len(task1_data)} task1 entries")

    # Get all unique keys (comment IDs) across all datasets
    all_keys = set(comments_data.keys()) | set(task1_data.keys())

    print(f"Merging data for {len(all_keys)} unique comment IDs...")

    # Create merged data list
    merged_data = []
    for key in sorted(all_keys):
        # Get data from each source
        comment_info = comments_data.get(key, {})
        task1_info = task1_data.get(key, {})

        # Create row
        row = {
            "document": comment_info.get("document", key[0]),
            "comment_id": comment_info.get("comment_id", key[1]),
            "comment": comment_info.get("comment", ""),
            "flausch": task1_info.get("flausch", ""),
        }
        merged_data.append(row)

    return merged_data


def process_test_data(comments_file):
    """
    Process test data (only comments, no labels).

    Returns:
        list: List of dictionaries containing test data
    """
    test_data = []

    print("Loading test comments...")
    with open(comments_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # For test data, we only have comments, no labels
            test_row = {
                "document": row["document"],
                "comment_id": row["comment_id"],
                "comment": row["comment"],
                "flausch": "",  # Empty column for consistency with training data
            }
            test_data.append(test_row)

    print(f"Loaded {len(test_data)} test comments")
    return test_data


def split_train_dev(data, train_ratio=0.8, random_seed=42):
    """
    Split data into train and dev sets.

    Args:
        data (list): List of data samples
        train_ratio (float): Ratio of data to use for training
        random_seed (int): Random seed for reproducibility

    Returns:
        tuple: (train_data, dev_data)
    """
    # Set random seed for reproducibility
    random.seed(random_seed)

    # Shuffle the data
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)

    # Calculate split point
    split_point = int(len(shuffled_data) * train_ratio)

    # Split the data
    train_data = shuffled_data[:split_point]
    dev_data = shuffled_data[split_point:]

    return train_data, dev_data


def write_csv_data(data, output_path):
    """
    Write data to CSV file.

    Args:
        data (list): List of dictionaries to write
        output_path (str): Path to output CSV file
    """
    if not data:
        print(f"Warning: No data to write to {output_path}")
        return

    # Determine fieldnames based on first row
    fieldnames = list(data[0].keys())

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)


def main():
    """Main function to handle command line arguments and execute merge."""

    parser = argparse.ArgumentParser(description="Merge FlauschErkennung CSV files and split into train/dev/test")
    parser.add_argument(
        "--training_data_dir",
        default="data/raw/FlauschErkennung/Data/training data",
        help="Directory containing training CSV files to merge",
    )
    parser.add_argument(
        "--test_data_dir",
        default="data/raw/FlauschErkennung/Data/test data",
        help="Directory containing test CSV files",
    )
    parser.add_argument(
        "--output_dir",
        default="data/Germeval/2025/FlauschErkennung/task1",
        help="Output directory for processed CSV files",
    )

    args = parser.parse_args()

    try:
        merge_flausch_data(args.training_data_dir, args.test_data_dir, args.output_dir)
    except Exception as e:
        print(f"Error during processing: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
