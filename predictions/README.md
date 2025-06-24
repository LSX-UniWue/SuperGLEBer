# GermEval 2025 Prediction Converter

This directory contains scripts to convert model predictions to the required submission formats for GermEval 2025 tasks.

## Overview

The main script `convert.py` processes prediction files from the `results/` directory and creates submission-ready files according to the specific format requirements of each task.

## Supported Tasks

- **Flausch Classification** (`flausch_class`): Binary classification (yes/no)
- **Flausch Tagging** (`flausch_tagging`): Named entity recognition with BIO tags
- **Harmful Content** (`harmful_c2a`, `harmful_dbo`, `harmful_vio`): Content classification
- **SustainEval Classification** (`sustaineval_class`): Multi-class classification
- **SustainEval Regression** (`sustaineval_regr`): Regression task
- **LLMs4Subjects** (`llms4subjects`): Multi-label domain classification

## Usage

### Basic Usage

```bash
# Convert all tasks for all models
python convert.py --task all

# Convert specific task for all models
python convert.py --task harmful_c2a

# Convert all tasks for a specific model
python convert.py --model LSX-UniWue_LLaMmlein_7B --task all

# Convert specific task for specific model
python convert.py --model LSX-UniWue_LLaMmlein_7B --task harmful_c2a
```

### Creating Submission Zip Files

To create submission-ready zip files according to the competition guidelines:

```bash
# Create submission zip files for all tasks
python convert.py --task all --create-zips

# Create submission zip files with custom team name and run number
python convert.py --task all --create-zips --team-name "MyTeam" --run-number 2

# Create submission zip files for specific tasks
python convert.py --task harmful_c2a --create-zips
python convert.py --task sustaineval_class --create-zips
python convert.py --task llms4subjects --create-zips
```

## Command Line Options

- `--task`: Specify which task(s) to convert
  - `flausch_class`: Flausch classification task
  - `flausch_tagging`: Flausch tagging task
  - `harmful_c2a`: Call-to-action detection
  - `harmful_dbo`: Democratic basic order attacks
  - `harmful_vio`: Violence detection
  - `sustaineval_class`: SustainEval classification
  - `sustaineval_regr`: SustainEval regression
  - `llms4subjects`: LLMs4Subjects domain classification
  - `all`: All available tasks (default)

- `--model`: Specify a single model to process (default: all models)

- `--create-zips`: Create submission zip files according to competition guidelines

- `--team-name`: Team name for submission files (default: "LSX-UniWue")

- `--run-number`: Run number for harmful content submissions, 1-3 (default: 1)

## Output Structure

The script creates files in the `converted/` directory:

```
converted/
├── [model_name]/
│   ├── Individual prediction files:
│   ├── task1-predicted.csv              # Flausch classification
│   ├── task2-predicted.csv              # Flausch tagging
│   ├── [model_name]_c2a.csv            # Harmful content c2a
│   ├── [model_name]_dbo.csv            # Harmful content dbo
│   ├── [model_name]_vio.csv            # Harmful content vio
│   ├── prediction_task_a.csv           # SustainEval classification
│   ├── prediction_task_b.csv           # SustainEval regression
│   ├── subtask_1/                      # LLMs4Subjects predictions
│   │
│   ├── Submission zip files (when --create-zips is used):
│   ├── [team_name][run_number].zip     # Harmful content submission
│   ├── [team_name]_[model]_sustaineval_submission.zip
│   ├── [team_name]_[model]_llms4subjects_submission.zip
│   └── [team_name]_[model]_flausch_submission.zip
```

## Submission Guidelines Compliance

The script creates submission files that comply with the official guidelines:

### Harmful Content

- Creates `[team_name][run].zip` containing `[team_name][run]_[task].csv` files
- Example: `LSX-UniWue1.zip` with `LSX-UniWue1_c2a.csv`, `LSX-UniWue1_dbo.csv`, `LSX-UniWue1_vio.csv`

### SustainEval

- Creates zip file with `prediction_task_a.csv`, `prediction_task_b.csv`, and `code/main.py`
- The `main.py` is a placeholder that should be replaced with actual prediction code

### LLMs4Subjects

- Creates zip file with `subtask_1/` and `subtask_2/` directories
- Follows the required folder structure with Article/Book/Conference/Report/Thesis subdirectories
- `subtask_2/` contains placeholder structure (empty directories)

### Flausch

- Creates zip file with `task1-predicted.csv` (classification) and `task2-predicted.csv` (tagging)

## Input Requirements

The script expects prediction files in `results/[model_name]/` with the following naming:

- `flausch_classification.tsv`
- `flausch_tagging.tsv`
- `harmful_content_c2a.tsv`
- `harmful_content_dbo.tsv`
- `harmful_content_vio.tsv`
- `sustaineval_classification.tsv`
- `sustaineval_regression.tsv`
- `llms4subjects.tsv`

## Notes

- The script automatically matches predictions with test data using normalized text comparison
- Missing predictions are filled with sensible defaults (e.g., "FALSE" for binary tasks, 0 for regression)
- Text normalization handles Unicode differences and emoji encoding variations
- For LLMs4Subjects, only subtask_1 is fully implemented; subtask_2 creates placeholder structure

## Examples

```bash
# Complete workflow for a single model
python convert.py --model LSX-UniWue_LLaMmlein_7B --task all --create-zips

# Create second run for harmful content with different team name
python convert.py --model LSX-UniWue_LLaMmlein_7B --task harmful_c2a --create-zips --team-name "MyTeam" --run-number 2

# Only process SustainEval tasks
python convert.py --task sustaineval_class --task sustaineval_regr --create-zips
```
