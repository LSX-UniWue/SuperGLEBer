#!/usr/bin/env python3
"""
Transform LLMs4Subjects training data into Flair-compatible format.
Extracts German records with domain labels, titles, and abstracts.
"""

import json
import os
import re
from pathlib import Path
from typing import List, Optional, Set, Tuple


def extract_domain_labels(subject_list: List[str]) -> Set[str]:
    """Extract domain codes from subject list that match linsearch:mapping pattern."""
    domains = set()
    for subject in subject_list:
        if isinstance(subject, str) and "(classificationName=linsearch:mapping)" in subject:
            # Extract the domain code after "linsearch:mapping)"
            match = re.search(r"\(classificationName=linsearch:mapping\)([a-zA-Z]+)", subject)
            if match:
                domains.add(match.group(1))
    return domains


def process_jsonld_file(file_path: Path) -> Optional[Tuple[Set[str], str, str]]:
    """
    Process a single JSON-LD file and extract domains, title, and abstract.
    Returns (domains, title, abstract) or None if processing fails.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Find the main record in the @graph
        main_record = None
        for item in data.get("@graph", []):
            if isinstance(item, dict) and "title" in item and "abstract" in item:
                main_record = item
                break

        if not main_record:
            return None

        # Extract domains from subject field
        subject_list = main_record.get("subject", [])
        domains = extract_domain_labels(subject_list)

        # Skip records without domain information
        if not domains:
            return None

        # Extract title and abstract
        title = main_record.get("title", "")
        abstract = main_record.get("abstract", "")

        # Handle cases where title/abstract might be lists
        if isinstance(title, list):
            title = " ".join(str(t) for t in title if t)
        else:
            title = str(title)

        if isinstance(abstract, list):
            abstract = " ".join(str(a) for a in abstract if a)
        else:
            abstract = str(abstract)

        title = title.strip()
        abstract = abstract.strip()

        # Skip records without title or abstract
        if not title or not abstract:
            return None

        return domains, title, abstract

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def transform_training_data(input_dir: Path, output_file: Path, language: str = "de"):
    """
    Transform LLMs4Subjects training data into Flair format.

    Args:
        input_dir: Path to the training data directory
        output_file: Path to output file
        language: Language code to process (default: "de")
    """
    records = []

    # Process all record types
    record_types = ["Article", "Book", "Conference", "Report", "Thesis"]

    for record_type in record_types:
        type_dir = input_dir / record_type / language
        if not type_dir.exists():
            print(f"Directory not found: {type_dir}")
            continue

        print(f"Processing {record_type} - {language}...")

        # Process all JSON-LD files in the directory
        for json_file in type_dir.glob("*.jsonld"):
            result = process_jsonld_file(json_file)
            if result:
                domains, title, abstract = result
                records.append((domains, title, abstract))

    print(f"Processed {len(records)} records total")

    # Write to output file in Flair format
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for domains, title, abstract in records:
            # Format domains as Flair labels
            domain_labels = " ".join([f"__label__{domain}" for domain in sorted(domains)])

            # Clean text (remove tabs and newlines)
            title_clean = title.replace("\t", " ").replace("\n", " ").replace("\r", " ")
            abstract_clean = abstract.replace("\t", " ").replace("\n", " ").replace("\r", " ")

            # Write in format: labels<tab>title<tab>abstract
            f.write(f"{domain_labels}\t{title_clean}\t{abstract_clean}\n")

    print(f"Output written to: {output_file}")

    # Print some statistics
    all_domains = set()
    for domains, _, _ in records:
        all_domains.update(domains)

    print(f"Found {len(all_domains)} unique domains: {sorted(all_domains)}")

    # Domain frequency analysis
    domain_counts = {}
    for domains, _, _ in records:
        for domain in domains:
            domain_counts[domain] = domain_counts.get(domain, 0) + 1

    print("\nDomain frequencies:")
    for domain, count in sorted(domain_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {domain}: {count}")


def main():
    # Paths
    script_dir = Path(__file__).parent
    base_dir = (
        script_dir.parent.parent.parent
        / "raw"
        / "llms4subjects-main"
        / "shared-task-datasets"
        / "TIBKAT"
        / "all-subjects"
        / "data"
    )

    # Process training data
    train_input_dir = base_dir / "train"
    train_output_file = script_dir / "train_de.txt"

    print("=" * 50)
    print("Processing Training Data")
    print("=" * 50)
    print(f"Input directory: {train_input_dir}")
    print(f"Output file: {train_output_file}")

    if train_input_dir.exists():
        transform_training_data(train_input_dir, train_output_file, language="de")
    else:
        print(f"Error: Training directory does not exist: {train_input_dir}")

    # Process dev data
    dev_input_dir = base_dir / "dev"
    dev_output_file = script_dir / "dev_de.txt"

    print("\n" + "=" * 50)
    print("Processing Development Data")
    print("=" * 50)
    print(f"Input directory: {dev_input_dir}")
    print(f"Output file: {dev_output_file}")

    if dev_input_dir.exists():
        transform_training_data(dev_input_dir, dev_output_file, language="de")
    else:
        print(f"Error: Development directory does not exist: {dev_input_dir}")

    # Process test data
    test_input_dir = base_dir / "test"
    test_output_file = script_dir / "test_de.txt"

    print("\n" + "=" * 50)
    print("Processing Test Data")
    print("=" * 50)
    print(f"Input directory: {test_input_dir}")
    print(f"Output file: {test_output_file}")

    if test_input_dir.exists():
        transform_training_data(test_input_dir, test_output_file, language="de")
    else:
        print(f"Error: Test directory does not exist: {test_input_dir}")


if __name__ == "__main__":
    main()
