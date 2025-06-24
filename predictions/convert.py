#!/usr/bin/env python3
"""
Script to convert model predictions to submission format for GermEval 2025 tasks.
"""

import argparse
import os
import re
import unicodedata
from pathlib import Path

import pandas as pd


def create_output_dir():
    """Create the converted output directory if it doesn't exist."""
    output_dir = Path("predictions/converted")
    output_dir.mkdir(exist_ok=True)
    return output_dir


def parse_flausch_predictions(prediction_file):
    """
    Parse flausch classification predictions from TSV format.

    Expected format:
    comment_text
     - Gold: yes/no
     - Pred: yes/no
     -> MISMATCH! (optional)
    """
    predictions = []
    current_comment = None
    current_prediction = None

    with open(prediction_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        # Skip empty lines
        if not line:
            # If we have a complete comment-prediction pair, save it
            if current_comment is not None and current_prediction is not None:
                predictions.append({"comment": current_comment, "prediction": current_prediction})
                current_comment = None
                current_prediction = None
            continue

        # Look for prediction line first
        if line.startswith("- Pred:"):
            pred_match = re.search(r"Pred:\s*(yes|no)", line)
            if pred_match:
                current_prediction = pred_match.group(1)
            continue

        # If line starts with "- " it's metadata, skip
        if line.startswith("- ") or line.startswith("->"):
            continue

        # Otherwise it's a comment
        # Save previous prediction if we have one
        if current_comment is not None and current_prediction is not None:
            predictions.append({"comment": current_comment, "prediction": current_prediction})

        current_comment = line
        current_prediction = None

    # Don't forget the last prediction if file doesn't end with empty line
    if current_comment is not None and current_prediction is not None:
        predictions.append({"comment": current_comment, "prediction": current_prediction})

    return predictions


def parse_harmful_predictions(prediction_file):
    """
    Parse harmful content predictions from TSV format.

    Expected format:
    comment_text
     - Gold: <label>
     - Pred: <label>
     -> MISMATCH! (optional)
    """
    predictions = []
    current_comment = None
    current_prediction = None

    with open(prediction_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        # Skip empty lines
        if not line:
            # If we have a complete comment-prediction pair, save it
            if current_comment is not None and current_prediction is not None:
                predictions.append({"comment": current_comment, "prediction": current_prediction})
                current_comment = None
                current_prediction = None
            continue

        # Look for prediction line first
        if line.startswith("- Pred:"):
            # Extract prediction value after "Pred:"
            pred_match = re.search(r"Pred:\s*(.+)", line)
            if pred_match:
                current_prediction = pred_match.group(1)
            continue

        # If line starts with "- " it's metadata, skip
        if line.startswith("- ") or line.startswith("->"):
            continue

        # Otherwise it's a comment
        # Save previous prediction if we have one
        if current_comment is not None and current_prediction is not None:
            predictions.append({"comment": current_comment, "prediction": current_prediction})

        current_comment = line
        current_prediction = None

    # Don't forget the last prediction if file doesn't end with empty line
    if current_comment is not None and current_prediction is not None:
        predictions.append({"comment": current_comment, "prediction": current_prediction})

    return predictions


def parse_sustaineval_classification_predictions(prediction_file):
    """
    Parse sustaineval classification predictions from TSV format.

    Expected format:
    context || target
     - Gold: <number>
     - Pred: <number>
     -> MISMATCH! (optional)
    """
    predictions = []
    current_comment = None
    current_prediction = None

    with open(prediction_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        # Skip empty lines
        if not line:
            # If we have a complete comment-prediction pair, save it
            if current_comment is not None and current_prediction is not None:
                predictions.append({"comment": current_comment, "prediction": current_prediction})
                current_comment = None
                current_prediction = None
            continue

        # Look for prediction line first
        if line.startswith("- Pred:"):
            # Extract prediction value after "Pred:"
            pred_match = re.search(r"Pred:\s*(\d+)", line)
            if pred_match:
                current_prediction = int(pred_match.group(1))
            continue

        # If line starts with "- " it's metadata, skip
        if line.startswith("- ") or line.startswith("->"):
            continue

        # Otherwise it's a comment (context || target format)
        # Save previous prediction if we have one
        if current_comment is not None and current_prediction is not None:
            predictions.append({"comment": current_comment, "prediction": current_prediction})

        current_comment = line
        current_prediction = None

    # Don't forget the last prediction if file doesn't end with empty line
    if current_comment is not None and current_prediction is not None:
        predictions.append({"comment": current_comment, "prediction": current_prediction})

    return predictions


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
        comment_id = row["comment_id"]

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
            "all",
        ],
        help="Task to convert predictions for (flausch_class: classification, flausch_tagging: sequence tagging)",
    )
    parser.add_argument("--model", help="Specific model to convert (if not provided, converts all models)")

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


if __name__ == "__main__":
    main()
