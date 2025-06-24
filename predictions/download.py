#!/usr/bin/env python3
"""
Download script for SuperGLEBer results from helma.nhr.fau.de
Downloads training logs and test files referenced in lookup.json
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote


def sanitize_filename(name):
    """Sanitize model name for use as directory name"""
    return name.replace("/", "_").replace(":", "_")


def download_file(remote_host, remote_path, local_path):
    """Download a file via scp"""
    try:
        # Create local directory if it doesn't exist
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Build scp command
        cmd = ["scp", f"{remote_host}:{remote_path}", str(local_path)]

        print(f"Downloading {remote_path} -> {local_path}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"✓ Successfully downloaded {local_path.name}")
            return True
        else:
            print(f"✗ Failed to download {remote_path}: {result.stderr}")
            return False

    except Exception as e:
        print(f"✗ Error downloading {remote_path}: {e}")
        return False


def find_test_files(remote_host, remote_dir):
    """Find test files in the remote directory"""
    try:
        # List files in remote directory
        cmd = ["ssh", remote_host, f"ls {remote_dir}/test.*"]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            return [line.strip() for line in result.stdout.split("\n") if line.strip()]
        else:
            # Try alternative locations
            alt_cmd = ["ssh", remote_host, f"find {remote_dir} -name 'test.*' -type f"]
            alt_result = subprocess.run(alt_cmd, capture_output=True, text=True)
            if alt_result.returncode == 0:
                return [line.strip() for line in alt_result.stdout.split("\n") if line.strip()]

        return []
    except Exception as e:
        print(f"Warning: Could not list test files in {remote_dir}: {e}")
        return []


def main():
    # Configuration
    remote_host = "helma.nhr.fau.de"
    lookup_file = Path("lookup.json")
    results_dir = Path("results")

    # Check if lookup.json exists
    if not lookup_file.exists():
        print(f"Error: {lookup_file} not found!")
        sys.exit(1)

    # Load lookup data
    try:
        with open(lookup_file, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {lookup_file}: {e}")
        sys.exit(1)

    print(f"Starting download from {remote_host}")
    print(f"Found {len(data)} models in lookup.json")
    print("-" * 50)

    total_files = 0
    successful_downloads = 0

    # Process each model
    for model_data in data:
        model_name = model_data["model"]
        sanitized_model = sanitize_filename(model_name)

        print(f"\nProcessing model: {model_name}")

        # Process each task for this model
        for task_name, task_info in model_data.items():
            if task_name == "model":
                continue

            if not isinstance(task_info, dict) or "train_log_path" not in task_info:
                continue

            remote_log_path = task_info["train_log_path"]

            # Create local paths
            model_dir = results_dir / sanitized_model
            local_log_path = model_dir / f"{task_name}.log"

            # Download training log
            total_files += 1
            if download_file(remote_host, remote_log_path, local_log_path):
                successful_downloads += 1

            # Find and download test files
            remote_dir = str(Path(remote_log_path).parent.parent)  # Go up from training_logs/training.log
            test_files = find_test_files(remote_host, remote_dir)

            for test_file in test_files:
                if test_file:
                    test_filename = Path(test_file).name
                    # Use task name + extension for local filename
                    file_ext = Path(test_file).suffix
                    local_test_path = model_dir / f"{task_name}{file_ext}"

                    total_files += 1
                    if download_file(remote_host, test_file, local_test_path):
                        successful_downloads += 1

    print("-" * 50)
    print(f"Download completed!")
    print(f"Successfully downloaded {successful_downloads}/{total_files} files")

    if successful_downloads < total_files:
        print("Some files failed to download. Check the output above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
