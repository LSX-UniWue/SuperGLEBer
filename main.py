#!/usr/bin/env python3
"""
Main script for SustainEval submission - Runs classification and/or regression tasks
This script reproduces the predictions for SustainEval subtasks.

Usage:
  python3 main.py                    # Run both tasks (default)
  python3 main.py --task both        # Run both tasks
  python3 main.py --task classification  # Run only classification (task A)
  python3 main.py --task regression      # Run only regression (task B)

Workflow:
1. Run training for the selected task(s)
2. Parse training output to find model paths
3. Convert predictions to submission format
4. Create separate submission zip files for each subtask (when running both)

Requirements:
- Python 3.10+
- All dependencies from requirements.txt
- Data in data/Germeval/2025/SustainEval/
- Source code in src/
"""

import argparse
import os
import re
import subprocess
import sys
import traceback
import zipfile
from pathlib import Path


def run_command_with_output(cmd, cwd=".", timeout=3600 * 24):
    """
    Run a command and capture both stdout and stderr with real-time output.

    Args:
        cmd: Command to run as list
        cwd: Working directory
        timeout: Timeout in seconds

    Returns:
        tuple: (returncode, stdout, stderr)
    """
    print(f"Running command: {' '.join(cmd)}")
    print(f"Working directory: {cwd}")
    print("-" * 60)

    try:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=cwd, bufsize=1, universal_newlines=True
        )

        stdout_lines = []
        stderr_lines = []

        # Read output in real-time
        while True:
            stdout_line = process.stdout.readline()
            stderr_line = process.stderr.readline()

            if stdout_line:
                print(f"STDOUT: {stdout_line.rstrip()}")
                stdout_lines.append(stdout_line)

            if stderr_line:
                print(f"STDERR: {stderr_line.rstrip()}")
                stderr_lines.append(stderr_line)

            if process.poll() is not None:
                break

        # Read any remaining output
        remaining_stdout, remaining_stderr = process.communicate()
        if remaining_stdout:
            stdout_lines.append(remaining_stdout)
            print(f"STDOUT: {remaining_stdout}")
        if remaining_stderr:
            stderr_lines.append(remaining_stderr)
            print(f"STDERR: {remaining_stderr}")

        returncode = process.returncode
        stdout = "".join(stdout_lines)
        stderr = "".join(stderr_lines)

        print("-" * 60)
        print(f"Command completed with return code: {returncode}")

        return returncode, stdout, stderr

    except subprocess.TimeoutExpired:
        print(f"Command timed out after {timeout} seconds")
        process.kill()
        return -1, "", "Command timed out"
    except Exception as e:
        print(f"Error running command: {e}")
        return -1, "", str(e)


def parse_model_path(stdout_text):
    """
    Parse the model training base path from training output.

    Args:
        stdout_text: Training stdout text

    Returns:
        str or None: Model base path if found
    """
    print("Parsing model path from training output...")

    # Look for the pattern: Model training base path: "/path/to/model"
    pattern = r'Model training base path: "([^"]+)"'
    match = re.search(pattern, stdout_text)

    if match:
        model_path = match.group(1)
        print(f"Found model training base path: {model_path}")
        return model_path
    else:
        print("Could not find model training base path in output")
        print("Searching for alternative patterns...")

        # Alternative patterns to try
        alt_patterns = [
            r"training_logs.*?([^\s]+/training_logs)",
            r"outputs/.*?/training_logs",
            r"Model.*?path.*?([^\s]+)",
        ]

        for pattern in alt_patterns:
            match = re.search(pattern, stdout_text, re.IGNORECASE)
            if match:
                alt_path = match.group(1)
                print(f"Found alternative path: {alt_path}")
                return alt_path

        return None


def find_test_predictions(model_path):
    """
    Find the test.tsv file in the model path.

    Args:
        model_path: Base path where model was trained

    Returns:
        str or None: Path to test.tsv if found
    """
    print(f"Looking for test predictions in: {model_path}")

    if not model_path:
        print("No model path provided")
        return None

    # Convert to Path object
    model_dir = Path(model_path)

    # Look for test.tsv in the model directory
    test_files = list(model_dir.glob("**/test.tsv"))

    if test_files:
        test_file = test_files[0]
        print(f"Found test predictions: {test_file}")
        return str(test_file)
    else:
        print("Could not find test.tsv in model directory")
        print("Searching in parent directories...")

        # Try parent directories
        parent_dirs = [model_dir.parent, model_dir.parent.parent]
        for parent in parent_dirs:
            test_files = list(parent.glob("**/test.tsv"))
            if test_files:
                test_file = test_files[0]
                print(f"Found test predictions in parent directory: {test_file}")
                return str(test_file)

        # Last resort: search entire root directory
        print("Searching entire root directory for test.tsv...")
        root_dir = Path(".")
        test_files = list(root_dir.glob("**/test.tsv"))
        if test_files:
            test_file = test_files[0]
            print(f"Found test predictions in root directory: {test_file}")
            return str(test_file)

        return None


def run_sustaineval_task(task_name, task_config):
    """
    Run a single SustainEval task (classification or regression).

    Args:
        task_name: Name of the task ("classification" or "regression")
        task_config: Task configuration name

    Returns:
        tuple: (success, model_path, test_file_path)
    """
    print(f"\n{'=' * 80}")
    print(f"RUNNING SUSTAINEVAL {task_name.upper()} TASK")
    print(f"{'=' * 80}")

    try:
        # Run training
        cmd = ["python3", "src/train.py", "+model=moderngbert_1B", f"+task={task_config}"]

        print(f"Starting {task_name} training...")
        returncode, stdout, stderr = run_command_with_output(cmd)

        if returncode != 0:
            print(f"Training failed with return code: {returncode}")
            print("STDERR:", stderr[-2000:])  # Last 2000 chars
            return False, None, None

        print(f"{task_name} training completed successfully!")

        # Parse model path
        model_path = parse_model_path(stdout)
        if not model_path:
            print(f"Could not parse model path for {task_name}")
            print("going to guess where the test.tsv file is")

        # Find test predictions
        test_file = find_test_predictions(model_path)
        if not test_file:
            print(f"Could not find test predictions for {task_name}")
            return False, model_path, None

        print(f"{task_name} task completed successfully!")
        print(f"Model path: {model_path}")
        print(f"Test file: {test_file}")

        return True, model_path, test_file

    except Exception as e:
        print(f"Error in {task_name} task: {e}")
        traceback.print_exc()
        return False, None, None


def convert_predictions(task_type, test_tsv_for_sustaineval):
    """
    Convert predictions to submission format.

    Args:
        task_type: Type of task ("sustaineval_class" or "sustaineval_regr")

    Returns:
        bool: Success status
    """
    print(f"\n{'=' * 60}")
    print(f"CONVERTING PREDICTIONS FOR {task_type}")
    print(f"{'=' * 60}")

    try:
        cmd = [
            "python3",
            "predictions/convert.py",
            "--task",
            task_type,
            "--create-zips",
            "--test_tsv_for_sustaineval",
            test_tsv_for_sustaineval,
        ]

        returncode, stdout, stderr = run_command_with_output(cmd)

        if returncode != 0:
            print(f"Conversion failed with return code: {returncode}")
            print("STDERR:", stderr[-1000:])
            return False

        print("Conversion completed successfully!")
        print("STDOUT:", stdout[-1000:])

        return True

    except Exception as e:
        print(f"Error converting predictions: {e}")
        traceback.print_exc()
        return False


def create_final_submission_zips():
    """
    Create final submission zip files for both subtasks.
    """
    print(f"\n{'=' * 60}")
    print("CREATING FINAL SUBMISSION ZIP FILES")
    print(f"{'=' * 60}")

    # Look for generated prediction files
    converted_dir = Path("predictions/converted")

    if not converted_dir.exists():
        print("No converted predictions directory found")
        return

    # Find model directories
    model_dirs = [d for d in converted_dir.iterdir() if d.is_dir()]

    for model_dir in model_dirs:
        print(f"Processing model directory: {model_dir.name}")

        # Look for existing zip files
        zip_files = list(model_dir.glob("*sustaineval*.zip"))

        if zip_files:
            print(f"Found {len(zip_files)} submission zip files:")
            for zip_file in zip_files:
                print(f"  - {zip_file.name}")

                # Verify zip contents
                try:
                    with zipfile.ZipFile(zip_file, "r") as zf:
                        files = zf.namelist()
                        print(f"    Contents: {files}")
                except Exception as e:
                    print(f"    Error reading zip: {e}")
        else:
            print("No submission zip files found")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="SustainEval Submission Script")
    parser.add_argument(
        "--task",
        choices=["classification", "regression", "both"],
        default="both",
        help="Which task to run (default: both)",
    )
    args = parser.parse_args()

    print("SustainEval Submission Script")
    print("=" * 80)
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Script location: {__file__}")
    print(f"Task selection: {args.task}")
    print("=" * 80)

    # Check prerequisites
    print("\nChecking prerequisites...")

    required_paths = [
        "src/train.py",
        "data/Germeval/2025/SustainEval/train.txt",
        "data/Germeval/2025/SustainEval/dev.txt",
        "data/Germeval/2025/SustainEval/test.txt",
        "predictions/convert.py",
        "requirements.txt",
    ]

    missing_paths = []
    for path in required_paths:
        if not Path(path).exists():
            missing_paths.append(path)
        else:
            print(f"✓ Found: {path}")

    if missing_paths:
        print("\n❌ Missing required files:")
        for path in missing_paths:
            print(f"  - {path}")
        print("\nPlease ensure all required files are present before running.")

    print("\n✓ Doesn't matter if we actually found all required files, we'll just run the tasks")

    # Track overall success
    overall_success = True
    completed_tasks = []

    # Task 1: Classification (run if task is "classification" or "both")
    if args.task in ["classification", "both"]:
        print(f"\n{'-' * 60}")
        print("TASK 1: SUSTAINEVAL CLASSIFICATION")
        print(f"{'-' * 60}")

        class_success, class_model_path, class_test_file = run_sustaineval_task(
            "classification", "sustaineval_classification"
        )

        if class_success:
            print("✓ Classification training completed")
            completed_tasks.append("classification")

            # Convert classification predictions
            if convert_predictions("sustaineval_class", class_test_file):
                print("✓ Classification predictions converted")
            else:
                print("❌ Classification prediction conversion failed")
                overall_success = False
        else:
            print("❌ Classification training failed")
            overall_success = False

    # Task 2: Regression (run if task is "regression" or "both")
    if args.task in ["regression", "both"]:
        print(f"\n{'-' * 60}")
        print("TASK 2: SUSTAINEVAL REGRESSION")
        print(f"{'-' * 60}")

        regr_success, regr_model_path, regr_test_file = run_sustaineval_task("regression", "sustaineval_regression")

        if regr_success:
            print("✓ Regression training completed")
            completed_tasks.append("regression")

            # Convert regression predictions
            if convert_predictions("sustaineval_regr", regr_test_file):
                print("✓ Regression predictions converted")
            else:
                print("❌ Regression prediction conversion failed")
                overall_success = False
        else:
            print("❌ Regression training failed")
            overall_success = False

    # Create final submission files (only if running both tasks)
    if args.task == "both":
        create_final_submission_zips()

    # Final summary
    print(f"\n{'=' * 80}")
    print("FINAL SUMMARY")
    print(f"{'=' * 80}")

    print(f"Completed tasks: {completed_tasks}")
    print(f"Overall success: {'✓' if overall_success else '❌'}")

    if overall_success:
        print("\n🎉 SustainEval submission completed successfully!")
        if args.task == "both":
            print("Check the predictions/converted/ directory for submission zip files.")
        else:
            print(f"Task '{args.task}' completed. Run with --task both to create submission zips.")
    else:
        print("\n⚠️  Some tasks failed. Check the output above for details.")
        print("You may need to manually complete the failed tasks.")

    # Show final directory structure (only if running both tasks)
    if args.task == "both":
        print(f"\n{'-' * 60}")
        print("FINAL DIRECTORY STRUCTURE")
        print(f"{'-' * 60}")

        converted_dir = Path("predictions/converted")
        if converted_dir.exists():
            for item in converted_dir.rglob("*"):
                if item.is_file():
                    rel_path = item.relative_to("predictions")
                    print(f"  {rel_path}")

    return 0 if overall_success else 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nScript interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)
