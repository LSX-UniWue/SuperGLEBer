#!/usr/bin/env python3
"""
Script to transform task2.csv and comments.csv into BIO labeled format.
Outputs exactly two columns: word and label.
Automatically splits training data into train/dev (80/20) and processes test data.
"""

import argparse
import csv
import os
import random
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


def process_training_data(task2_path: str, comments_path: str) -> List[Tuple[str, str]]:
    """
    Process training data and convert to BIO format.

    Returns:
        List of (token, label) tuples
    """
    # Read comments
    print("Reading training comments...")
    comments_dict = read_comments(comments_path)

    # Read task2 annotations
    print("Reading training annotations...")
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
            bio_data.append((token, label))

        # Add empty line between comments (sentence boundary)
        if tokens:  # Only add if there were tokens
            bio_data.append(("", ""))

        processed_count += 1
        if processed_count % 1000 == 0:
            print(f"Processed {processed_count} training comments...")

    print(f"Done! Processed {processed_count} training comments.")
    return bio_data


def process_test_data(comments_path: str) -> List[Tuple[str, str]]:
    """
    Process test data (only comments, no annotations).

    Returns:
        List of (token, label) tuples where labels are all "O"
    """
    # Read test comments
    print("Reading test comments...")
    comments_dict = read_comments(comments_path)

    # Process each comment
    bio_data = []
    processed_count = 0

    for key, comment_text in comments_dict.items():
        # Tokenize the comment
        tokens = tokenize_text(comment_text)

        # For test data, all labels are "O" (no annotations available)
        for token, start_pos, end_pos in tokens:
            bio_data.append((token, "O"))

        # Add empty line between comments (sentence boundary)
        if tokens:  # Only add if there were tokens
            bio_data.append(("", ""))

        processed_count += 1
        if processed_count % 1000 == 0:
            print(f"Processed {processed_count} test comments...")

    print(f"Done! Processed {processed_count} test comments.")
    return bio_data


def split_bio_data_by_sentences(
    bio_data: List[Tuple[str, str]], train_ratio=0.8, random_seed=42
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Split BIO data into train and dev sets by sentences.

    Args:
        bio_data: List of (token, label) tuples
        train_ratio: Ratio of sentences to use for training
        random_seed: Random seed for reproducibility

    Returns:
        tuple: (train_data, dev_data)
    """
    # Set random seed for reproducibility
    random.seed(random_seed)

    # Split data into sentences (groups separated by empty lines)
    sentences = []
    current_sentence = []

    for token, label in bio_data:
        if token == "" and label == "":
            # End of sentence
            if current_sentence:
                sentences.append(current_sentence)
                current_sentence = []
        else:
            current_sentence.append((token, label))

    # Add the last sentence if it doesn't end with empty line
    if current_sentence:
        sentences.append(current_sentence)

    print(f"Found {len(sentences)} sentences in training data")

    # Shuffle sentences
    random.shuffle(sentences)

    # Calculate split point
    split_point = int(len(sentences) * train_ratio)

    # Split sentences
    train_sentences = sentences[:split_point]
    dev_sentences = sentences[split_point:]

    # Flatten back to token list format
    train_data = []
    for sentence in train_sentences:
        train_data.extend(sentence)
        train_data.append(("", ""))  # Add sentence boundary

    dev_data = []
    for sentence in dev_sentences:
        dev_data.extend(sentence)
        dev_data.append(("", ""))  # Add sentence boundary

    return train_data, dev_data


def write_bio_data(bio_data: List[Tuple[str, str]], output_path: str):
    """
    Write BIO data to TSV file.

    Args:
        bio_data: List of (token, label) tuples
        output_path: Path to output TSV file
    """
    print(f"Writing to {output_path}...")

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        for token, label in bio_data:
            writer.writerow([token, label])


def process_files(training_task2_path: str, training_comments_path: str, test_comments_path: str, output_dir: str):
    """
    Main processing function to convert CSV files to BIO format.
    Splits training data into train/dev and processes test data separately.
    """
    # Process training data
    print("Processing training data...")
    training_bio_data = process_training_data(training_task2_path, training_comments_path)

    # Process test data
    print("Processing test data...")
    test_bio_data = process_test_data(test_comments_path)

    # Split training data into train/dev
    print("Splitting training data into train/dev (80/20)...")
    train_data, dev_data = split_bio_data_by_sentences(training_bio_data, train_ratio=0.8)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Write training split
    train_path = os.path.join(output_dir, "train.tsv")
    write_bio_data(train_data, train_path)
    print(f"Training data saved to: {train_path} ({len([x for x in train_data if x[0] != ''])} tokens)")

    # Write development split
    dev_path = os.path.join(output_dir, "dev.tsv")
    write_bio_data(dev_data, dev_path)
    print(f"Development data saved to: {dev_path} ({len([x for x in dev_data if x[0] != ''])} tokens)")

    # Write test data
    test_path = os.path.join(output_dir, "test.tsv")
    write_bio_data(test_bio_data, test_path)
    print(f"Test data saved to: {test_path} ({len([x for x in test_bio_data if x[0] != ''])} tokens)")

    print("Processing complete!")


def main():
    """Main function to handle command line arguments and execute processing."""

    parser = argparse.ArgumentParser(
        description="Transform FlauschErkennung data to BIO format and split into train/dev/test"
    )
    parser.add_argument(
        "--training_task2_path",
        default="data/raw/FlauschErkennung/Data/training data/task2.csv",
        help="Path to training task2.csv file",
    )
    parser.add_argument(
        "--training_comments_path",
        default="data/raw/FlauschErkennung/Data/training data/comments.csv",
        help="Path to training comments.csv file",
    )
    parser.add_argument(
        "--test_comments_path",
        default="data/raw/FlauschErkennung/Data/test data/comments.csv",
        help="Path to test comments.csv file",
    )
    parser.add_argument(
        "--output_dir",
        default="data/Germeval/2025/FlauschErkennung/task2",
        help="Output directory for processed TSV files",
    )

    args = parser.parse_args()

    try:
        process_files(args.training_task2_path, args.training_comments_path, args.test_comments_path, args.output_dir)
    except Exception as e:
        print(f"Error during processing: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
