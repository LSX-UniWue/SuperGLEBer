#!/usr/bin/env python3
"""
Script to convert model predictions to submission format for GermEval 2025 tasks.
"""

import argparse
import glob
import json
import os
import re
import unicodedata
import zipfile
from pathlib import Path

import pandas as pd


def create_output_dir():
    """Create the converted output directory if it doesn't exist."""
    output_dir = Path("predictions/converted")
    output_dir.mkdir(exist_ok=True)
    return output_dir


def parse_predictions_generic(prediction_file, content_key="comment", prediction_processor=None):
    """
    Generic function to parse predictions from TSV format.

    Expected format:
    content_text
     - Gold: <label>
     - Pred: <label>
     -> MISMATCH! (optional)

    Args:
        prediction_file: Path to the prediction file
        content_key: Key name for the content in returned dict ("comment", "abstract", etc.)
        prediction_processor: Function to process the raw prediction string

    Returns:
        List of dicts with content_key and "prediction" keys
    """
    predictions = []
    current_content = None
    current_prediction = None

    with open(prediction_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        # Skip empty lines
        if not line:
            # If we have a complete content-prediction pair, save it
            if current_content is not None and current_prediction is not None:
                predictions.append({content_key: current_content, "prediction": current_prediction})
                current_content = None
                current_prediction = None
            continue

        # Look for prediction line first
        if line.startswith("- Pred:"):
            # Extract prediction value after "Pred:"
            pred_match = re.search(r"Pred:\s*(.+)", line)
            if pred_match:
                pred_text = pred_match.group(1).strip()
                # Apply custom processor if provided
                if prediction_processor:
                    current_prediction = prediction_processor(pred_text)
                else:
                    current_prediction = pred_text
            continue

        # If line starts with "- " it's metadata, skip
        if line.startswith("- ") or line.startswith("->"):
            continue

        # Otherwise it's content (comment/abstract/etc.)
        # Save previous prediction if we have one
        if current_content is not None and current_prediction is not None:
            predictions.append({content_key: current_content, "prediction": current_prediction})

        current_content = line
        current_prediction = None

    # Don't forget the last prediction if file doesn't end with empty line
    if current_content is not None and current_prediction is not None:
        predictions.append({content_key: current_content, "prediction": current_prediction})

    return predictions


def parse_flausch_predictions(prediction_file):
    """Parse flausch classification predictions (yes/no format)."""

    def processor(pred_text):
        pred_match = re.search(r"(yes|no)", pred_text)
        return pred_match.group(1) if pred_match else "no"

    return parse_predictions_generic(prediction_file, "comment", processor)


def parse_harmful_predictions(prediction_file):
    """Parse harmful content predictions (general label format)."""
    return parse_predictions_generic(prediction_file, "comment")


def parse_sustaineval_classification_predictions(prediction_file):
    """Parse sustaineval classification predictions (numeric format)."""

    def processor(pred_text):
        pred_match = re.search(r"(\d+)", pred_text)
        return int(pred_match.group(1)) if pred_match else 0

    return parse_predictions_generic(prediction_file, "comment", processor)


def parse_sustaineval_regression_predictions(prediction_file):
    """
    Parse sustaineval regression predictions from TSV format.

    Expected format:
    context\ttarget\tgold_score\tpredicted_score
    """
    predictions = []

    with open(prediction_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Split by tab
        parts = line.split("\t")
        if len(parts) >= 4:
            context = parts[0]
            target = parts[1]
            # gold_score = float(parts[2])  # We don't need this for predictions
            predicted_score = float(parts[3])

            # Combine context and target to match test data format
            comment = f"{context} || {target}"
            predictions.append({"comment": comment, "prediction": predicted_score})

    return predictions


def parse_flausch_tagging_predictions(prediction_file):
    """
    Parse flausch tagging predictions from BIO format.

    Expected format:
    word gold_label predicted_label
    word gold_label predicted_label
    (empty line indicates new comment)
    """
    predictions = []
    current_tokens = []
    current_predictions = []

    with open(prediction_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        # Empty line indicates end of comment
        if not line:
            if current_tokens and current_predictions:
                # Reconstruct comment text and extract entities
                comment_text = " ".join(current_tokens)
                entities = extract_entities_from_bio(current_tokens, current_predictions)
                predictions.append({"comment": comment_text, "entities": entities})

            current_tokens = []
            current_predictions = []
            continue

        # Parse token line
        parts = line.split()
        if len(parts) >= 3:
            token = parts[0]
            # gold_label = parts[1]  # We don't need this for predictions
            pred_label = parts[2]

            current_tokens.append(token)
            current_predictions.append(pred_label)

    # Don't forget the last comment if file doesn't end with empty line
    if current_tokens and current_predictions:
        comment_text = " ".join(current_tokens)
        entities = extract_entities_from_bio(current_tokens, current_predictions)
        predictions.append({"comment": comment_text, "entities": entities})

    return predictions


def extract_entities_from_bio(tokens, bio_labels):
    """
    Extract entities from BIO labels and calculate character positions.

    Returns list of entities with type, start, and end positions.
    """
    entities = []
    current_entity = None
    current_start = 0
    current_pos = 0

    for i, (token, label) in enumerate(zip(tokens, bio_labels)):
        if label.startswith("B-"):
            # End previous entity if exists
            if current_entity is not None:
                entities.append(
                    {
                        "type": current_entity["type"],
                        "start": current_entity["start"],
                        "end": current_pos - 1,  # -1 to not include the space
                    }
                )

            # Start new entity
            entity_type = label[2:]  # Remove "B-" prefix
            current_entity = {"type": entity_type, "start": current_pos}
        elif label.startswith("I-") and current_entity is not None:
            # Continue current entity - nothing special to do
            pass
        elif label == "O":
            # End current entity if exists
            if current_entity is not None:
                entities.append(
                    {
                        "type": current_entity["type"],
                        "start": current_entity["start"],
                        "end": current_pos - 1,  # -1 to not include the space
                    }
                )
                current_entity = None

        # Update character position
        current_pos += len(token)
        if i < len(tokens) - 1:  # Add space except for last token
            current_pos += 1

    # End final entity if exists
    if current_entity is not None:
        entities.append({"type": current_entity["type"], "start": current_entity["start"], "end": current_pos})

    return entities


def normalize_text(text):
    """Normalize text for better matching by handling emoji encoding differences."""
    if pd.isna(text):
        return "<EMPTY-COMMENT>"

    # Convert to string and normalize Unicode
    text = str(text)

    # Remove emoji variation selectors (U+FE0F) that cause mismatches
    text = text.replace("\ufe0f", "")

    # Normalize Unicode to NFC form
    text = unicodedata.normalize("NFC", text)

    # Normalize newlines and multiple spaces to single spaces
    # This handles the case where test data has \n but predictions have spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def load_test_data(test_file, separator=None):
    """Load the test data with document and comment IDs."""
    if separator is None:
        # Auto-detect separator based on filename
        if "HarmfulContent" in test_file:
            separator = ";"
        else:
            separator = ","
    return pd.read_csv(test_file, sep=separator)


def convert_task(model_name, output_dir, task_config):
    """Generic function to convert predictions to submission format."""

    prediction_file = task_config["prediction_file"]
    test_file = task_config["test_file"]
    parser_func = task_config["parser_func"]
    test_text_column = task_config["test_text_column"]
    id_columns = task_config["id_columns"]
    output_column = task_config["output_column"]
    output_file = task_config["output_file"]
    default_prediction = task_config["default_prediction"]
    prediction_converter = task_config.get("prediction_converter")

    if not os.path.exists(prediction_file):
        print(f"Warning: Prediction file not found: {prediction_file}")
        return

    if not os.path.exists(test_file):
        print(f"Error: Test file not found: {test_file}")
        return

    print(f"Converting {task_config['task_name']} predictions for model: {model_name}")

    # Parse predictions and normalize text
    predictions = parser_func(prediction_file)
    print(f"Found {len(predictions)} predictions")

    # Normalize prediction comments
    for pred in predictions:
        pred["comment_normalized"] = normalize_text(pred["comment"])

    # Load test data
    test_data = load_test_data(test_file, task_config.get("separator"))
    print(f"Found {len(test_data)} test samples")

    # Match predictions with test data by comment text
    submission_data = []
    matched_count = 0

    for idx, row in test_data.iterrows():
        test_comment = normalize_text(row[test_text_column])

        # Extract ID columns
        id_data = {col: row[col] for col in id_columns}
        if "comment_id" in id_data:
            # convert all comment_id to int
            id_data["comment_id"] = int(id_data["comment_id"])

        # Find matching prediction
        prediction = None
        for pred in predictions:
            if pred["comment_normalized"] == test_comment:
                prediction = pred["prediction"]
                matched_count += 1
                break

        if prediction is None:
            # Only print warning for first few unmatched comments to avoid spam
            if len([x for x in submission_data if x.get(output_column) is None]) < 10:
                print(f"Warning: No prediction found for comment: {str(test_comment)[:50]}...")
            prediction = default_prediction

        # Apply prediction conversion if provided
        if prediction_converter and prediction is not None:
            prediction = prediction_converter(prediction)

        # Create submission entry
        submission_entry = {**id_data, output_column: prediction}
        submission_data.append(submission_entry)

    print(f"Successfully matched {matched_count}/{len(test_data)} predictions")

    # Create submission DataFrame
    submission_df = pd.DataFrame(submission_data)

    # Save to CSV
    model_output_dir = output_dir / model_name
    model_output_dir.mkdir(exist_ok=True)

    output_path = model_output_dir / output_file
    submission_df.to_csv(output_path, index=False)

    print(f"Saved submission file: {output_path}")
    print(f"Submission contains {len(submission_df)} entries")

    # Print sample of results
    print("\nSample predictions:")
    print(submission_df.head(10))


def convert_flausch_task(model_name, output_dir):
    """Convert flausch classification predictions to submission format."""

    task_config = {
        "task_name": "flausch classification",
        "prediction_file": f"predictions/results/{model_name}/flausch_classification.tsv",
        "test_file": "data/Germeval/2025/FlauschErkennung/task1/test.csv",
        "parser_func": parse_flausch_predictions,
        "test_text_column": "comment",
        "id_columns": ["document", "comment_id"],
        "output_column": "flausch",
        "output_file": "task1-predicted.csv",
        "default_prediction": "no",
        "separator": ",",
    }

    convert_task(model_name, output_dir, task_config)


def convert_harmful_task(model_name, output_dir, task_type):
    """Convert harmful content predictions to submission format."""

    # Task mapping
    task_mapping = {
        "c2a": {
            "prediction_file": f"predictions/results/{model_name}/harmful_content_c2a.tsv",
            "test_file": "data/Germeval/2025/HarmfulContent/c2a/test.csv",
            "column_name": "c2a",
            "output_file": f"{model_name}_c2a.csv",
            "default_prediction": "FALSE",
        },
        "dbo": {
            "prediction_file": f"predictions/results/{model_name}/harmful_content_dbo.tsv",
            "test_file": "data/Germeval/2025/HarmfulContent/dbo/test.csv",
            "column_name": "dbo",
            "output_file": f"{model_name}_dbo.csv",
            "default_prediction": "nothing",
        },
        "vio": {
            "prediction_file": f"predictions/results/{model_name}/harmful_content_vio.tsv",
            "test_file": "data/Germeval/2025/HarmfulContent/vio/test.csv",
            "column_name": "vio",
            "output_file": f"{model_name}_vio.csv",
            "default_prediction": "FALSE",
        },
    }

    if task_type not in task_mapping:
        print(f"Error: Unknown task type: {task_type}")
        return

    task_info = task_mapping[task_type]

    def harmful_prediction_converter(prediction):
        """Convert prediction format for harmful content tasks."""
        if prediction is None:
            return None

        if task_type == "c2a" or task_type == "vio":
            # Convert True/False to TRUE/FALSE
            if prediction.lower() == "true":
                return "TRUE"
            elif prediction.lower() == "false":
                return "FALSE"
        return prediction

    task_config = {
        "task_name": f"harmful content {task_type}",
        "prediction_file": task_info["prediction_file"],
        "test_file": task_info["test_file"],
        "parser_func": parse_harmful_predictions,
        "test_text_column": "description",
        "id_columns": ["id"],
        "output_column": task_info["column_name"],
        "output_file": task_info["output_file"],
        "default_prediction": task_info["default_prediction"],
        "separator": ";",
        "prediction_converter": harmful_prediction_converter,
    }

    convert_task(model_name, output_dir, task_config)


def convert_sustaineval_classification_task(model_name, output_dir):
    """Convert sustaineval classification predictions to submission format."""

    task_config = {
        "task_name": "sustaineval classification",
        "prediction_file": f"predictions/results/{model_name}/sustaineval_classification.tsv",
        "test_file": "data/Germeval/2025/SustainEval/test.txt",
        "parser_func": parse_sustaineval_classification_predictions,
        "test_text_column": "combined_text",  # We'll create this column
        "id_columns": ["id"],
        "output_column": "label",
        "output_file": "prediction_task_a.csv",
        "default_prediction": 0,
        "separator": "\t",
    }

    # We need a custom function to handle the context||target combination
    def custom_convert_sustaineval_classification(model_name, output_dir, task_config):
        prediction_file = task_config["prediction_file"]
        test_file = task_config["test_file"]
        parser_func = task_config["parser_func"]
        id_columns = task_config["id_columns"]
        output_column = task_config["output_column"]
        output_file = task_config["output_file"]
        default_prediction = task_config["default_prediction"]

        if not os.path.exists(prediction_file):
            print(f"Warning: Prediction file not found: {prediction_file}")
            return

        if not os.path.exists(test_file):
            print(f"Error: Test file not found: {test_file}")
            return

        print(f"Converting {task_config['task_name']} predictions for model: {model_name}")

        # Parse predictions and normalize text
        predictions = parser_func(prediction_file)
        print(f"Found {len(predictions)} predictions")

        # Normalize prediction comments
        for pred in predictions:
            pred["comment_normalized"] = normalize_text(pred["comment"])

        # Load test data manually to handle context||target combination
        test_data = pd.read_csv(test_file, sep="\t")
        print(f"Found {len(test_data)} test samples")

        # Create combined text column for matching
        test_data["combined_text"] = test_data["context"] + " || " + test_data["target"]

        # Match predictions with test data by combined text
        submission_data = []
        matched_count = 0

        for idx, row in test_data.iterrows():
            test_comment = normalize_text(row["combined_text"])

            # Extract ID
            id_data = {col: row[col] for col in id_columns}

            # Find matching prediction
            prediction = None
            for pred in predictions:
                if pred["comment_normalized"] == test_comment:
                    prediction = pred["prediction"]
                    matched_count += 1
                    break

            if prediction is None:
                if len([x for x in submission_data if x.get(output_column) is None]) < 10:
                    print(f"Warning: No prediction found for comment: {str(test_comment)[:50]}...")
                prediction = default_prediction

            # Create submission entry
            submission_entry = {**id_data, output_column: prediction}
            submission_data.append(submission_entry)

        print(f"Successfully matched {matched_count}/{len(test_data)} predictions")

        # Create submission DataFrame
        submission_df = pd.DataFrame(submission_data)

        # Save to CSV
        model_output_dir = output_dir / model_name
        model_output_dir.mkdir(exist_ok=True)

        output_path = model_output_dir / output_file
        submission_df.to_csv(output_path, index=False)

        print(f"Saved submission file: {output_path}")
        print(f"Submission contains {len(submission_df)} entries")

        # Print sample of results
        print("\nSample predictions:")
        print(submission_df.head(10))

    custom_convert_sustaineval_classification(model_name, output_dir, task_config)


def convert_sustaineval_regression_task(model_name, output_dir):
    """Convert sustaineval regression predictions to submission format."""

    task_config = {
        "task_name": "sustaineval regression",
        "prediction_file": f"predictions/results/{model_name}/sustaineval_regression.tsv",
        "test_file": "data/Germeval/2025/SustainEval/test.txt",
        "parser_func": parse_sustaineval_regression_predictions,
        "test_text_column": "combined_text",  # We'll create this column
        "id_columns": ["id"],
        "output_column": "label",
        "output_file": "prediction_task_b.csv",
        "default_prediction": 0.0,
        "separator": "\t",
    }

    # We need a custom function to handle the context||target combination
    def custom_convert_sustaineval_regression(model_name, output_dir, task_config):
        prediction_file = task_config["prediction_file"]
        test_file = task_config["test_file"]
        parser_func = task_config["parser_func"]
        id_columns = task_config["id_columns"]
        output_column = task_config["output_column"]
        output_file = task_config["output_file"]
        default_prediction = task_config["default_prediction"]

        if not os.path.exists(prediction_file):
            print(f"Warning: Prediction file not found: {prediction_file}")
            return

        if not os.path.exists(test_file):
            print(f"Error: Test file not found: {test_file}")
            return

        print(f"Converting {task_config['task_name']} predictions for model: {model_name}")

        # Parse predictions and normalize text
        predictions = parser_func(prediction_file)
        print(f"Found {len(predictions)} predictions")

        # Normalize prediction comments
        for pred in predictions:
            pred["comment_normalized"] = normalize_text(pred["comment"])

        # Load test data manually to handle context||target combination
        test_data = pd.read_csv(test_file, sep="\t")
        print(f"Found {len(test_data)} test samples")

        # Create combined text column for matching
        test_data["combined_text"] = test_data["context"] + " || " + test_data["target"]

        # Match predictions with test data by combined text
        submission_data = []
        matched_count = 0

        for idx, row in test_data.iterrows():
            test_comment = normalize_text(row["combined_text"])

            # Extract ID
            id_data = {col: row[col] for col in id_columns}

            # Find matching prediction
            prediction = None
            for pred in predictions:
                if pred["comment_normalized"] == test_comment:
                    prediction = pred["prediction"]
                    matched_count += 1
                    break

            if prediction is None:
                if len([x for x in submission_data if x.get(output_column) is None]) < 10:
                    print(f"Warning: No prediction found for comment: {str(test_comment)[:50]}...")
                prediction = default_prediction

            # Create submission entry
            submission_entry = {**id_data, output_column: prediction}
            submission_data.append(submission_entry)

        print(f"Successfully matched {matched_count}/{len(test_data)} predictions")

        # Create submission DataFrame
        submission_df = pd.DataFrame(submission_data)

        # Save to CSV
        model_output_dir = output_dir / model_name
        model_output_dir.mkdir(exist_ok=True)

        output_path = model_output_dir / output_file
        submission_df.to_csv(output_path, index=False)

        print(f"Saved submission file: {output_path}")
        print(f"Submission contains {len(submission_df)} entries")

        # Print sample of results
        print("\nSample predictions:")
        print(submission_df.head(10))

    custom_convert_sustaineval_regression(model_name, output_dir, task_config)


def convert_flausch_tagging_task(model_name, output_dir):
    """Convert flausch tagging predictions to submission format."""

    prediction_file = f"predictions/results/{model_name}/flausch_tagging.tsv"
    test_file = "data/Germeval/2025/FlauschErkennung/task2/test_comments.csv"

    if not os.path.exists(prediction_file):
        print(f"Warning: Prediction file not found: {prediction_file}")
        return

    if not os.path.exists(test_file):
        print(f"Error: Test file not found: {test_file}")
        return

    print(f"Converting flausch tagging predictions for model: {model_name}")

    # Parse predictions and normalize text
    predictions = parse_flausch_tagging_predictions(prediction_file)
    print(f"Found {len(predictions)} predictions")

    # Normalize prediction comments
    for pred in predictions:
        pred["comment_normalized"] = normalize_text(pred["comment"])

    # Load test data
    test_data = pd.read_csv(test_file)
    print(f"Found {len(test_data)} test samples")

    # Create lookup table from predictions by normalized comment text
    prediction_lookup = {}
    for pred in predictions:
        prediction_lookup[pred["comment_normalized"]] = pred["entities"]

    # Match test data with predictions using lookup table
    submission_data = []
    matched_count = 0
    unmatched_count = 0

    for idx, row in test_data.iterrows():
        test_comment = normalize_text(row["comment"])
        document = row["document"]
        comment_id = int(row["comment_id"])

        # Find matching prediction using lookup
        matched_entities = prediction_lookup.get(test_comment, [])

        if matched_entities:
            matched_count += 1
        else:
            unmatched_count += 1
            # Only print warning for first few unmatched comments to avoid spam
            if unmatched_count <= 10:
                print(f"Warning: No prediction found for comment: {str(test_comment)[:50]}...")

        # Add all entities for this comment to submission data
        for entity in matched_entities:
            submission_entry = {
                "document": document,
                "comment_id": comment_id,
                "type": entity["type"],
                "start": entity["start"],
                "end": entity["end"],
            }
            submission_data.append(submission_entry)

    print(f"Successfully matched {matched_count}/{len(test_data)} predictions")

    # Create submission DataFrame
    submission_df = pd.DataFrame(submission_data)

    # Save to CSV
    model_output_dir = output_dir / model_name
    model_output_dir.mkdir(exist_ok=True)

    output_path = model_output_dir / "task2-predicted.csv"
    submission_df.to_csv(output_path, index=False)

    print(f"Saved submission file: {output_path}")
    print(f"Submission contains {len(submission_df)} entries")

    # Print sample of results
    print("\nSample predictions:")
    print(submission_df.head(10))


def parse_llms4subjects_predictions(prediction_file):
    """Parse llms4subjects predictions (comma-separated domains format)."""

    def processor(pred_text):
        if pred_text:
            domains = [d.strip() for d in pred_text.split(",") if d.strip()]
            return domains
        else:
            return []

    return parse_predictions_generic(prediction_file, "abstract", processor)


def convert_llms4subjects_task(model_name, output_dir):
    """Convert llms4subjects predictions to submission format."""

    prediction_file = f"predictions/results/{model_name}/llms4subjects.tsv"
    test_base_dir = "data/Germeval/2025/llms4subjects/test"

    if not os.path.exists(prediction_file):
        print(f"Warning: Prediction file not found: {prediction_file}")
        return

    if not os.path.exists(test_base_dir):
        print(f"Error: Test directory not found: {test_base_dir}")
        return

    print(f"Converting llms4subjects predictions for model: {model_name}")

    # Parse predictions and normalize text
    predictions = parse_llms4subjects_predictions(prediction_file)
    print(f"Found {len(predictions)} predictions")

    # Normalize prediction abstracts
    for pred in predictions:
        pred["abstract_normalized"] = normalize_text(pred["abstract"])

    # Create lookup table from predictions by normalized abstract text
    prediction_lookup = {}
    for pred in predictions:
        prediction_lookup[pred["abstract_normalized"]] = pred["prediction"]

    # Create output directory structure
    model_output_dir = output_dir / model_name
    model_output_dir.mkdir(exist_ok=True)

    # Create subtask_1 directory structure matching test data
    subtask_dir = model_output_dir / "subtask_1"
    subtask_dir.mkdir(exist_ok=True)

    # Find all test files and process them
    test_files = glob.glob(f"{test_base_dir}/**/*.jsonld", recursive=True)
    print(f"Found {len(test_files)} test files")

    matched_count = 0
    unmatched_count = 0

    for test_file_path in test_files:
        # Get relative path from test_base_dir
        rel_path = os.path.relpath(test_file_path, test_base_dir)

        # Create corresponding output directory structure
        output_file_dir = subtask_dir / os.path.dirname(rel_path)
        output_file_dir.mkdir(parents=True, exist_ok=True)

        # Change file extension from .jsonld to .json
        output_filename = os.path.basename(rel_path).replace(".jsonld", ".json")
        output_file_path = output_file_dir / output_filename

        # Load and parse the test file
        try:
            with open(test_file_path, "r", encoding="utf-8") as f:
                test_data = json.load(f)

            # Extract abstract from the test data
            abstract = None
            if "@graph" in test_data and len(test_data["@graph"]) > 0:
                for item in test_data["@graph"]:
                    if "abstract" in item:
                        abstract_data = item["abstract"]
                        # Handle case where abstract is a list
                        if isinstance(abstract_data, list):
                            # Take the first abstract if it's a list
                            abstract = abstract_data[0] if abstract_data else None
                        else:
                            abstract = abstract_data
                        break

            if abstract is None:
                print(f"Warning: No abstract found in {test_file_path}")
                prediction = []
            else:
                # Normalize abstract and look up prediction
                normalized_abstract = normalize_text(abstract)
                prediction = prediction_lookup.get(normalized_abstract, [])

                if prediction:
                    matched_count += 1
                else:
                    unmatched_count += 1
                    # Only print warning for first few unmatched files to avoid spam
                    if unmatched_count <= 10:
                        print(f"Warning: No prediction found for file: {rel_path}")

            # Create output JSON in the required format
            output_data = {"domains": prediction}

            # Save the prediction file
            with open(output_file_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=4, ensure_ascii=False)

        except Exception as e:
            print(f"Error processing file {test_file_path}: {e}")
            continue

    print(f"Successfully matched {matched_count}/{len(test_files)} predictions")
    print(f"Saved predictions to: {subtask_dir}")

    # Create a simple zip file for submission (optional)
    submission_zip_path = model_output_dir / f"{model_name}_llms4subjects.zip"

    # Note: You may want to create the zip file manually or use a different approach
    print(f"Conversion complete. You can zip the subtask_1 directory for submission.")
    print(f"Subtask directory: {subtask_dir}")


def create_harmful_content_submission_zip(output_dir, model_name, team_name="LSX-UniWue", run_number=1):
    """
    Create submission zip file for harmful content tasks according to guidelines.

    Format: [team_name][run].zip containing [team_name][run]_[task].csv files
    """
    model_output_dir = output_dir / model_name

    # Check which harmful content files exist
    harmful_tasks = ["c2a", "dbo", "vio"]
    available_files = []

    for task in harmful_tasks:
        csv_file = model_output_dir / f"{model_name}_{task}.csv"
        if csv_file.exists():
            available_files.append((task, csv_file))

    if not available_files:
        print(f"No harmful content prediction files found for {model_name}")
        return

    # Create submission zip file
    zip_filename = f"{team_name}{run_number}.zip"
    zip_path = model_output_dir / zip_filename

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for task, csv_file in available_files:
            # Create submission filename according to guidelines
            submission_filename = f"{team_name}{run_number}_{task}.csv"
            zipf.write(csv_file, submission_filename)
            print(f"Added {csv_file.name} as {submission_filename} to zip")

    print(f"Created harmful content submission zip: {zip_path}")
    print(f"Contains {len(available_files)} task predictions")


def create_sustaineval_submission_zip(output_dir, model_name, team_name="LSX-UniWue"):
    """
    Create submission zip file for sustaineval tasks according to guidelines.

    Should contain prediction_task_a.csv and prediction_task_b.csv
    """
    model_output_dir = output_dir / model_name

    # Check which sustaineval files exist
    task_a_file = model_output_dir / "prediction_task_a.csv"
    task_b_file = model_output_dir / "prediction_task_b.csv"

    available_files = []
    if task_a_file.exists():
        available_files.append(("prediction_task_a.csv", task_a_file))
    if task_b_file.exists():
        available_files.append(("prediction_task_b.csv", task_b_file))

    if not available_files:
        print(f"No sustaineval prediction files found for {model_name}")
        return

    # Create submission zip file
    zip_filename = f"{team_name}_{model_name}_sustaineval.zip"
    zip_path = model_output_dir / zip_filename

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for submission_filename, file_path in available_files:
            zipf.write(file_path, submission_filename)
            print(f"Added {file_path.name} as {submission_filename} to zip")

        # Create a basic code folder with main.py (for evaluation phase)
        # This is a placeholder - users should replace with actual code
        main_py_content = """#!/usr/bin/env python3
# Placeholder main.py for SustainEval submission
# Replace this with your actual prediction code

def main():
    print("This is a placeholder main.py file")
    print("Replace with your actual prediction code for evaluation phase")

if __name__ == "__main__":
    main()
"""
        zipf.writestr("code/main.py", main_py_content)
        print("Added placeholder code/main.py to zip")

    print(f"Created sustaineval submission zip: {zip_path}")
    print(f"Contains {len(available_files)} task predictions + code folder")


def create_llms4subjects_submission_zip(output_dir, model_name, team_name="LSX-UniWue"):
    """
    Create submission zip file for llms4subjects tasks according to guidelines.

    Should contain subtask_1 and subtask_2 folders (currently only subtask_1 is implemented)
    """
    model_output_dir = output_dir / model_name
    subtask1_dir = model_output_dir / "subtask_1"

    if not subtask1_dir.exists():
        print(f"No llms4subjects predictions found for {model_name}")
        return

    # Create submission zip file
    zip_filename = f"{team_name}_{model_name}_llms4subjects.zip"
    zip_path = model_output_dir / zip_filename

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        # Add all files from subtask_1 directory
        for root, dirs, files in os.walk(subtask1_dir):
            for file in files:
                file_path = Path(root) / file
                # Calculate relative path from model_output_dir
                rel_path = file_path.relative_to(model_output_dir)
                zipf.write(file_path, str(rel_path))

        # Create empty subtask_2 directory structure (placeholder)
        # Since only subtask_1 is implemented, create basic structure for subtask_2
        subtask2_base_dirs = [
            "subtask_2/Article/en/",
            "subtask_2/Article/de/",
            "subtask_2/Book/",
            "subtask_2/Conference/",
            "subtask_2/Report/",
            "subtask_2/Thesis/",
        ]

        for dir_path in subtask2_base_dirs:
            # Create empty directories by adding a .gitkeep file
            zipf.writestr(f"{dir_path}.gitkeep", "")

    print(f"Created llms4subjects submission zip: {zip_path}")
    print(f"Contains subtask_1 predictions and placeholder subtask_2 structure")


def create_flausch_submission_zip(output_dir, model_name, team_name="LSX-UniWue"):
    """
    Create separate submission zip files for each flausch subtask.

    Creates separate zips for:
    - task1-predicted.csv (classification)
    - task2-predicted.csv (tagging)

    Each zip contains only one CSV file as per submission guidelines.
    """
    model_output_dir = output_dir / model_name

    # Check which flausch files exist and create separate zips
    task1_file = model_output_dir / "task1-predicted.csv"
    task2_file = model_output_dir / "task2-predicted.csv"

    created_zips = []

    # Create zip for task1 (classification) if it exists
    if task1_file.exists():
        zip_filename = f"{team_name}_{model_name}_flausch_task1.zip"
        zip_path = model_output_dir / zip_filename

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(task1_file, "task1-predicted.csv")
            print(f"Added {task1_file.name} as task1-predicted.csv to zip")

        print(f"Created flausch task1 (classification) submission zip: {zip_path}")
        created_zips.append("task1")

    # Create zip for task2 (tagging) if it exists
    if task2_file.exists():
        zip_filename = f"{team_name}_{model_name}_flausch_task2.zip"
        zip_path = model_output_dir / zip_filename

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(task2_file, "task2-predicted.csv")
            print(f"Added {task2_file.name} as task2-predicted.csv to zip")

        print(f"Created flausch task2 (tagging) submission zip: {zip_path}")
        created_zips.append("task2")

    if not created_zips:
        print(f"No flausch prediction files found for {model_name}")
        return

    print(f"Created {len(created_zips)} separate flausch submission zip files: {', '.join(created_zips)}")


def main():
    parser = argparse.ArgumentParser(description="Convert model predictions to submission format")
    parser.add_argument(
        "--task",
        default="all",
        choices=[
            "flausch_class",
            "flausch_tagging",
            "harmful_c2a",
            "harmful_dbo",
            "harmful_vio",
            "sustaineval_class",
            "sustaineval_regr",
            "llms4subjects",
            "all",
        ],
        help="Task to convert predictions for (flausch_class: classification, flausch_tagging: sequence tagging)",
    )
    parser.add_argument("--model", help="Specific model to convert (if not provided, converts all models)")
    parser.add_argument("--team-name", default="LSX-UniWue", help="Team name for submission files")
    parser.add_argument("--run-number", type=int, default=1, help="Run number for harmful content submissions (1-3)")
    parser.add_argument("--create-zips", action="store_true", help="Create submission zip files")

    args = parser.parse_args()

    # Create output directory
    output_dir = create_output_dir()

    # Get list of available models
    results_dir = Path("predictions/results")
    if not results_dir.exists():
        print("Error: predictions/results directory not found")
        return

    available_models = [d.name for d in results_dir.iterdir() if d.is_dir()]

    if args.model:
        if args.model not in available_models:
            print(f"Error: Model {args.model} not found in results directory")
            print(f"Available models: {available_models}")
            return
        models_to_process = [args.model]
    else:
        models_to_process = available_models

    print(f"Processing models: {models_to_process}")

    # Process each model
    for model_name in models_to_process:
        print(f"\n{'=' * 50}")
        print(f"Processing model: {model_name}")
        print(f"{'=' * 50}")

        if args.task == "flausch_class" or args.task == "all":
            convert_flausch_task(model_name, output_dir)

        if args.task == "flausch_tagging" or args.task == "all":
            convert_flausch_tagging_task(model_name, output_dir)

        if args.task == "harmful_c2a" or args.task == "all":
            convert_harmful_task(model_name, output_dir, "c2a")

        if args.task == "harmful_dbo" or args.task == "all":
            convert_harmful_task(model_name, output_dir, "dbo")

        if args.task == "harmful_vio" or args.task == "all":
            convert_harmful_task(model_name, output_dir, "vio")

        if args.task == "sustaineval_class" or args.task == "all":
            convert_sustaineval_classification_task(model_name, output_dir)

        if args.task == "sustaineval_regr" or args.task == "all":
            convert_sustaineval_regression_task(model_name, output_dir)

        if args.task == "llms4subjects" or args.task == "all":
            convert_llms4subjects_task(model_name, output_dir)

        # Create submission zip files if requested
        if args.create_zips:
            print(f"\n{'=' * 30}")
            print(f"Creating submission zip files for {model_name}")
            print(f"{'=' * 30}")

            # Create harmful content submission zip
            if any(args.task in ["harmful_c2a", "harmful_dbo", "harmful_vio", "all"] for _ in [1]):
                create_harmful_content_submission_zip(output_dir, model_name, args.team_name, args.run_number)

            # Create sustaineval submission zip
            if any(args.task in ["sustaineval_class", "sustaineval_regr", "all"] for _ in [1]):
                create_sustaineval_submission_zip(output_dir, model_name, args.team_name)

            # Create llms4subjects submission zip
            if args.task == "llms4subjects" or args.task == "all":
                create_llms4subjects_submission_zip(output_dir, model_name, args.team_name)

            # Create flausch submission zip
            if any(args.task in ["flausch_class", "flausch_tagging", "all"] for _ in [1]):
                create_flausch_submission_zip(output_dir, model_name, args.team_name)


if __name__ == "__main__":
    main()
