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


def combine_csv_files(file1, file2, output_file):
    """
    Combine two CSV files into one.

    Args:
        file1 (str): Path to the first CSV file
        file2 (str): Path to the second CSV file
        output_file (str): Path to the output combined CSV file
    """
    with open(output_file, "w", encoding="utf-8", newline="") as outfile:
        writer = None

        # Process first file
        with open(file1, "r", encoding="utf-8") as infile1:
            reader1 = csv.DictReader(infile1)
            writer = csv.DictWriter(outfile, fieldnames=reader1.fieldnames)
            writer.writeheader()

            for row in reader1:
                writer.writerow(row)

        # Process second file (skip header)
        with open(file2, "r", encoding="utf-8") as infile2:
            reader2 = csv.DictReader(infile2)
            for row in reader2:
                writer.writerow(row)

    print(f"Combined {file1} and {file2} into {output_file}")


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

    # Define file mappings and special handling
    file_mappings = {}
    validation_files = []

    # Convert each JSONL file to CSV with special handling
    for jsonl_file in jsonl_files:
        base_name = os.path.basename(jsonl_file)

        # Skip trial files
        if "trial" in base_name.lower():
            print(f"Skipping trial file: {base_name}")
            continue

        # Handle file naming
        if "validation" in base_name.lower():
            csv_file = os.path.join(dest_dir, "validation_temp.csv")
            validation_files.append(csv_file)
        elif "development" in base_name.lower():
            csv_file = os.path.join(dest_dir, "development_temp.csv")
            validation_files.append(csv_file)
        elif "evaluation" in base_name.lower():
            csv_file = os.path.join(dest_dir, "test.csv")
        else:
            # Keep original name for training and other files
            csv_file = os.path.join(dest_dir, base_name.replace(".jsonl", ".csv"))

        convert_jsonl_to_csv(jsonl_file, csv_file)

    # Combine validation and development files into a single validation file
    if len(validation_files) == 2:
        final_validation_file = os.path.join(dest_dir, "validation.csv")
        combine_csv_files(validation_files[0], validation_files[1], final_validation_file)

        # Clean up temporary files
        for temp_file in validation_files:
            os.remove(temp_file)
    elif len(validation_files) == 1:
        # If only one validation file, just rename it
        final_validation_file = os.path.join(dest_dir, "validation.csv")
        os.rename(validation_files[0], final_validation_file)

    print("Conversion complete!")


if __name__ == "__main__":
    main()
