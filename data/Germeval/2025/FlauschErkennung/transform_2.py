#!/usr/bin/env python3
"""
Script to transform task2.csv and comments.csv into BIO labeled format.
Outputs exactly two columns: word and label.
"""

import csv
import os
import re
from typing import Dict, List, Tuple


def tokenize_text(text: str) -> List[Tuple[str, int, int]]:
    """
    Simple tokenization that preserves character positions.
    Returns list of (token, start_pos, end_pos) tuples.
    """
    tokens = []
    # Use regex to find word tokens and their positions
    for match in re.finditer(r"\S+", text):
        token = match.group()
        start = match.start()
        end = match.end()
        tokens.append((token, start, end))
    return tokens


def get_bio_label(token_start: int, token_end: int, entities: List[Dict]) -> str:
    """
    Determine BIO label for a token based on entity annotations.
    """
    for entity in entities:
        entity_start = int(entity["start"])
        entity_end = int(entity["end"])
        entity_type = entity["type"].replace(" ", "_")  # Replace spaces with underscores

        # Check if token overlaps with entity
        if token_start < entity_end and token_end > entity_start:
            # Token overlaps with entity
            if token_start == entity_start:
                # Beginning of entity
                return f"B-{entity_type}"
            else:
                # Inside entity
                return f"I-{entity_type}"

    return "O"  # Outside any entity


def read_comments(comments_path: str) -> Dict[Tuple[str, str], str]:
    """
    Read comments CSV and return a dictionary mapping (document, comment_id) to comment text.
    """
    comments_dict = {}
    with open(comments_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["document"], str(row["comment_id"]))
            comments_dict[key] = row["comment"]
    return comments_dict


def read_annotations(task2_path: str) -> Dict[Tuple[str, str], List[Dict]]:
    """
    Read task2 CSV and return a dictionary mapping (document, comment_id) to list of entities.
    """
    annotations_dict = {}
    with open(task2_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["document"], str(row["comment_id"]))
            if key not in annotations_dict:
                annotations_dict[key] = []
            annotations_dict[key].append({"type": row["type"], "start": int(row["start"]), "end": int(row["end"])})
    return annotations_dict


def process_files(task2_path: str, comments_path: str, output_dir: str):
    """
    Main processing function to convert CSV files to BIO format.
    """
    # Read comments
    print("Reading comments...")
    comments_dict = read_comments(comments_path)

    # Read task2 annotations
    print("Reading annotations...")
    annotations_dict = read_annotations(task2_path)

    # Process each comment
    bio_data = []
    processed_count = 0

    for key, comment_text in comments_dict.items():
        if key in annotations_dict:
            entities = annotations_dict[key]
        else:
            entities = []  # No annotations for this comment

        # Tokenize the comment
        tokens = tokenize_text(comment_text)

        # Assign BIO labels
        for token, start_pos, end_pos in tokens:
            label = get_bio_label(start_pos, end_pos, entities)
            bio_data.append([token, label])

        # Add empty line between comments (sentence boundary)
        if tokens:  # Only add if there were tokens
            bio_data.append(["", ""])

        processed_count += 1
        if processed_count % 1000 == 0:
            print(f"Processed {processed_count} comments...")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Write to output file
    output_path = os.path.join(output_dir, "flausch_bio.tsv")
    print(f"Writing to {output_path}...")

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        for row in bio_data:
            writer.writerow(row)

    print(f"Done! Processed {processed_count} comments.")
    print(f"Output saved to: {output_path}")


def main():
    # File paths
    task2_path = "../../../raw/FlauschErkennung/Data/training data/task2.csv"
    comments_path = "../../../raw/FlauschErkennung/Data/training data/comments.csv"
    output_dir = "."  # Current directory (FlauschErkennung)

    process_files(task2_path, comments_path, output_dir)


if __name__ == "__main__":
    main()
