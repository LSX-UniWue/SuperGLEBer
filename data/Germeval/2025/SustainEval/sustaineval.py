import csv
import glob
import json
import os
from pathlib import Path


def convert_jsonl_to_csv(input_file, output_file):
    """
    Convert a JSONL file to CSV format.

    Args:
        input_file (str): Path to the input JSONL file
        output_file (str): Path to the output CSV file
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Read JSONL file and convert to CSV
    with (
        open(input_file, "r", encoding="utf-8") as jsonl_file,
        open(output_file, "w", encoding="utf-8", newline="") as csv_file,
    ):
        # Read the first line to get the field names
        first_line = jsonl_file.readline().strip()
        if not first_line:
            print(f"Warning: {input_file} is empty")
            return

        # Parse the first JSON object to get field names
        fields = list(json.loads(first_line).keys())

        # Create CSV writer
        csv_writer = csv.DictWriter(csv_file, fieldnames=fields)
        csv_writer.writeheader()

        # Write the first line
        json_obj = json.loads(first_line)
        if type(json_obj["context"]) == list:
            json_obj["context"] = " ".join(json_obj["context"])
        if type(json_obj["target"]) == list:
            json_obj["target"] = " ".join(json_obj["target"])
        csv_writer.writerow(json_obj)

        # Process the rest of the file
        for line in jsonl_file:
            line = line.strip()
            if line:  # Skip empty lines
                json_obj = json.loads(line)
                if type(json_obj["context"]) == list:
                    json_obj["context"] = " ".join(json_obj["context"])
                if type(json_obj["target"]) == list:
                    json_obj["target"] = " ".join(json_obj["target"])
                csv_writer.writerow(json_obj)

    print(f"Converted {input_file} to {output_file}")


def main():
    # Source and destination directories
    source_dir = "data/raw/sustaineval2025_data-main"
    dest_dir = "data/Germeval/2025/SustainEval"

    # Create destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)

    # Find all JSONL files in the source directory
    jsonl_files = glob.glob(os.path.join(source_dir, "*.jsonl"))

    if not jsonl_files:
        print(f"No JSONL files found in {source_dir}")
        return

    print(f"Found {len(jsonl_files)} JSONL files to convert")

    # Convert each JSONL file to CSV
    for jsonl_file in jsonl_files:
        base_name = os.path.basename(jsonl_file)
        csv_file = os.path.join(dest_dir, base_name.replace(".jsonl", ".csv"))
        convert_jsonl_to_csv(jsonl_file, csv_file)

    print("Conversion complete!")


if __name__ == "__main__":
    main()
