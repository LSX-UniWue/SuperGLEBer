#!/usr/bin/env python3
"""
Script to split overall.txt into train, dev, and test sets with balanced multilabel distribution.
Maintains 70:10:20 ratio while ensuring all labels are present in each split.
"""

import argparse
import os
import random
import re
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple


def parse_line(line: str) -> Tuple[List[str], str]:
    """
    Parse a line to extract labels and text.

    Args:
        line: Input line from the dataset

    Returns:
        Tuple of (labels_list, text)
    """
    line = line.strip()

    # Find all labels using regex
    label_pattern = r"__label__([^__\t]+)"
    labels = re.findall(label_pattern, line)

    # Extract text (everything after the last label)
    # Split by tab and take the last part as text
    parts = line.split("\t")
    if len(parts) >= 2:
        text = "\t".join(parts[1:])  # Join in case text contains tabs
    else:
        # Fallback: remove all label patterns
        text = re.sub(r"__label__[^__\t]+\s*", "", line).strip()

    return labels, text


def get_label_signature(labels: List[str]) -> str:
    """
    Create a signature for a set of labels for stratification.

    Args:
        labels: List of labels for an example

    Returns:
        String signature representing the label combination
    """
    return "|".join(sorted(labels))


def stratified_split(
    data: List[Tuple[List[str], str]],
    train_ratio: float = 0.7,
    dev_ratio: float = 0.1,
    test_ratio: float = 0.2,
    random_seed: int = 42,
) -> Tuple[List, List, List]:
    """
    Perform stratified split on multilabel data.

    Args:
        data: List of (labels, text) tuples
        train_ratio: Proportion for training set
        dev_ratio: Proportion for development set
        test_ratio: Proportion for test set
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_data, dev_data, test_data)
    """
    random.seed(random_seed)

    # Group data by label signature
    signature_groups = defaultdict(list)
    for i, (labels, text) in enumerate(data):
        signature = get_label_signature(labels)
        signature_groups[signature].append((labels, text, i))

    train_data, dev_data, test_data = [], [], []

    # For each label combination, split proportionally
    for signature, examples in signature_groups.items():
        random.shuffle(examples)
        n_examples = len(examples)

        # Calculate split sizes
        n_train = max(1, int(n_examples * train_ratio))
        n_dev = max(1, int(n_examples * dev_ratio)) if n_examples > 1 else 0
        n_test = n_examples - n_train - n_dev

        # Ensure at least one example in test if we have enough examples
        if n_test == 0 and n_examples > 2:
            n_train -= 1
            n_test = 1

        # Split the examples
        train_examples = examples[:n_train]
        dev_examples = examples[n_train : n_train + n_dev]
        test_examples = examples[n_train + n_dev :]

        # Add to respective splits (remove index)
        train_data.extend([(labels, text) for labels, text, _ in train_examples])
        dev_data.extend([(labels, text) for labels, text, _ in dev_examples])
        test_data.extend([(labels, text) for labels, text, _ in test_examples])

    # Shuffle the final splits
    random.shuffle(train_data)
    random.shuffle(dev_data)
    random.shuffle(test_data)

    return train_data, dev_data, test_data


def format_line(labels: List[str], text: str) -> str:
    """
    Format labels and text back into the original format.

    Args:
        labels: List of labels
        text: Text content

    Returns:
        Formatted line string
    """
    label_str = " ".join([f"__label__{label}" for label in labels])
    return f"{label_str}\t{text}"


def analyze_splits(train_data: List, dev_data: List, test_data: List) -> None:
    """
    Analyze and print statistics about the splits.

    Args:
        train_data: Training data
        dev_data: Development data
        test_data: Test data
    """

    def get_label_stats(data):
        label_counts = Counter()
        signature_counts = Counter()

        for labels, _ in data:
            signature = get_label_signature(labels)
            signature_counts[signature] += 1
            for label in labels:
                label_counts[label] += 1

        return label_counts, signature_counts

    train_labels, train_sigs = get_label_stats(train_data)
    dev_labels, dev_sigs = get_label_stats(dev_data)
    test_labels, test_sigs = get_label_stats(test_data)

    print(f"\n=== SPLIT ANALYSIS ===")
    print(f"Total examples: {len(train_data) + len(dev_data) + len(test_data)}")
    print(
        f"Train: {len(train_data)} ({len(train_data) / (len(train_data) + len(dev_data) + len(test_data)) * 100:.1f}%)"
    )
    print(f"Dev:   {len(dev_data)} ({len(dev_data) / (len(train_data) + len(dev_data) + len(test_data)) * 100:.1f}%)")
    print(f"Test:  {len(test_data)} ({len(test_data) / (len(train_data) + len(dev_data) + len(test_data)) * 100:.1f}%)")

    print(f"\n=== LABEL DISTRIBUTION ===")
    all_labels = set(train_labels.keys()) | set(dev_labels.keys()) | set(test_labels.keys())

    for label in sorted(all_labels):
        train_count = train_labels.get(label, 0)
        dev_count = dev_labels.get(label, 0)
        test_count = test_labels.get(label, 0)
        total = train_count + dev_count + test_count

        print(
            f"{label:20s}: Train={train_count:4d} ({train_count / total * 100:5.1f}%), "
            f"Dev={dev_count:3d} ({dev_count / total * 100:5.1f}%), "
            f"Test={test_count:3d} ({test_count / total * 100:5.1f}%)"
        )

    print(f"\n=== LABEL COMBINATION STATISTICS ===")
    print(f"Unique combinations - Train: {len(train_sigs)}, Dev: {len(dev_sigs)}, Test: {len(test_sigs)}")

    # Check coverage
    all_signatures = set(train_sigs.keys()) | set(dev_sigs.keys()) | set(test_sigs.keys())
    train_only = set(train_sigs.keys()) - set(dev_sigs.keys()) - set(test_sigs.keys())
    if train_only:
        print(f"Combinations only in train: {len(train_only)}")

    common_sigs = set(train_sigs.keys()) & set(dev_sigs.keys()) & set(test_sigs.keys())
    print(
        f"Combinations in all splits: {len(common_sigs)}/{len(all_signatures)} ({len(common_sigs) / len(all_signatures) * 100:.1f}%)"
    )


def main():
    parser = argparse.ArgumentParser(description="Split multilabel hotel aspect dataset")
    parser.add_argument("--input", default="overall.txt", help="Input file path")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train split ratio")
    parser.add_argument("--dev-ratio", type=float, default=0.1, help="Dev split ratio")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", default=".", help="Output directory")

    args = parser.parse_args()

    # Validate ratios
    total_ratio = args.train_ratio + args.dev_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")

    print(f"Loading data from {args.input}...")

    # Load and parse data
    data = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                labels, text = parse_line(line)
                if not labels:
                    print(f"Warning: No labels found in line {line_num}")
                    continue
                if not text.strip():
                    print(f"Warning: No text found in line {line_num}")
                    continue

                data.append((labels, text))
            except Exception as e:
                print(f"Error parsing line {line_num}: {e}")
                continue

    print(f"Loaded {len(data)} examples")

    # Analyze label distribution before splitting
    print(f"\n=== ORIGINAL DATASET ANALYSIS ===")
    all_labels = Counter()
    all_signatures = Counter()

    for labels, _ in data:
        signature = get_label_signature(labels)
        all_signatures[signature] += 1
        for label in labels:
            all_labels[label] += 1

    print(f"Total examples: {len(data)}")
    print(f"Unique labels: {len(all_labels)}")
    print(f"Unique label combinations: {len(all_signatures)}")

    print("\nMost common label combinations:")
    for sig, count in all_signatures.most_common(10):
        print(f"  {sig}: {count} examples")

    # Perform stratified split
    print(f"\nPerforming stratified split...")
    train_data, dev_data, test_data = stratified_split(
        data, args.train_ratio, args.dev_ratio, args.test_ratio, args.seed
    )

    # Analyze splits
    analyze_splits(train_data, dev_data, test_data)

    # Write splits to files
    splits = [("train.txt", train_data), ("dev.txt", dev_data), ("test.txt", test_data)]

    for filename, split_data in splits:
        filepath = os.path.join(args.output_dir, filename)
        print(f"\nWriting {len(split_data)} examples to {filepath}")

        with open(filepath, "w", encoding="utf-8") as f:
            for labels, text in split_data:
                f.write(format_line(labels, text) + "\n")

    print("\nSplit completed successfully!")


if __name__ == "__main__":
    main()
