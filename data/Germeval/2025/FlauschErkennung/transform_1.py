#!/usr/bin/env python3
"""
Script to join two CSV files from FlauschErkennung data on comment_id
and create a merged dataset with all columns.
"""

import argparse
import csv
import os


def merge_flausch_data(data_dir, output_dir):
    """
    Merge the two FlauschErkennung CSV files on comment_id.

    Args:
        data_dir (str): Path to directory containing the two CSV files
        output_dir (str): Path to output directory for merged file
    """

    # File paths
    comments_file = os.path.join(data_dir, "comments.csv")
    task1_file = os.path.join(data_dir, "task1.csv")

    # Check if all files exist
    for file_path in [comments_file, task1_file]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required file not found: {file_path}")

    print("Loading CSV files...")

    # Data structures to hold the data
    comments_data = {}  # key: (document, comment_id), value: comment data
    task1_data = {}  # key: (document, comment_id), value: task1 data

    # Load comments.csv
    print("Loading comments...")
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

    print(f"Loaded {len(comments_data)} comments")
    print(f"Loaded {len(task1_data)} task1 entries")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Determine output filename based on input directory
    if "training" in data_dir.lower():
        output_filename = "merged_training_data.csv"
    elif "trial" in data_dir.lower():
        output_filename = "merged_trial_data.csv"
    else:
        output_filename = "merged_data.csv"

    output_path = os.path.join(output_dir, output_filename)

    # Get all unique keys (comment IDs) across all datasets
    all_keys = set(comments_data.keys()) | set(task1_data.keys())

    print(f"Merging data for {len(all_keys)} unique comment IDs...")

    # Write merged data
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        # Define the output columns
        fieldnames = ["document", "comment_id", "comment", "flausch"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        row_count = 0
        for key in sorted(all_keys):
            # Get data from each source
            comment_info = comments_data.get(key, {})
            task1_info = task1_data.get(key, {})

            # Create one row per comment
            row = {
                "document": comment_info.get("document", key[0]),
                "comment_id": comment_info.get("comment_id", key[1]),
                "comment": comment_info.get("comment", ""),
                "flausch": task1_info.get("flausch", ""),
            }
            writer.writerow(row)
            row_count += 1

    print(f"Saving merged data to: {output_path}")
    print(f"Final merged dataset has {row_count} rows")
    print("Merge complete!")
    print(f"Columns in merged dataset: {fieldnames}")


def main():
    """Main function to handle command line arguments and execute merge."""

    parser = argparse.ArgumentParser(description="Merge FlauschErkennung CSV files on comment_id")
    parser.add_argument(
        "--data_dir",
        default="data/raw/FlauschErkennung/Data/training data",
        help="Directory containing the two CSV files to merge",
    )
    parser.add_argument(
        "--output_dir", default="data/Germeval/2025/FlauschErkennung", help="Output directory for merged CSV file"
    )

    args = parser.parse_args()

    try:
        merge_flausch_data(args.data_dir, args.output_dir)
    except Exception as e:
        print(f"Error during merge: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
