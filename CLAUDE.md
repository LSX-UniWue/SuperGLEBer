# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

SuperGLEBer is a comprehensive Natural Language Understanding benchmark suite for German language evaluation. It consists of 29 different NLU tasks spanning document classification, sequence tagging, sentence similarity, and question answering. The codebase trains and evaluates various German language models on these tasks.

## Core Architecture

The project uses **Hydra** for configuration management and supports three main training frameworks:

1. **Flair-based training** (`train_flair.py`): For sequence tagging and text classification tasks using Flair embeddings
2. **Hugging Face QA training** (`train_hf_qa.py`): For question answering tasks using HF transformers
3. **Sentence Transformers training** (`train_st.py`): For sentence similarity and embedding-based tasks

### Configuration System

Hydra configuration is organized in `src/conf/` with these main groups:
- **config.yaml**: Base configuration (sets defaults for train_args, train_procedure)
- **model/**: Model definitions (40+ pre-configured German models from various sources)
- **task/**: Task definitions with corpus type, embedding type, and framework selection
- **train_args/**: Hardware-specific training parameters (a100, h100, gtx1080ti, rtx2080ti)
- **train_procedure/**: Training strategies (qlora, peft configs, bitsandbytes quantization settings)

### Key Configuration Classes

See `src/config.py` for dataclasses that define the structure. The main entry point merges YAML configs into a `SuperGLEBerConfig` object passed to training functions.

## Common Development Commands

**Run a training experiment** (most common task):
```bash
python src/train.py +model=gbert_base +train_args=a100 +task=news_class
```

Override individual config values:
```bash
python src/train.py +model=gbert_base +train_args=a100 +task=news_class train_args.batch_size=8
```

**Generate k8s/SLURM submission files**:
```bash
python src/template_k8s.py
```
(Note: This requires editing user-specific paths in the template file)

**Parse evaluation results**:
- Flair outputs: `src/evaluation/parse_flair_outputs.ipynb`
- Generic outputs: `src/evaluation/parse_outputs.ipynb`

## Important Patterns & Architecture Details

### Data Handling
- All data-related corpus loading is delegated to Flair library (ColumnCorpus, CSVClassificationCorpus, etc.)
- Data folder paths in task configs are relative to `data_base_dir` from base config
- Task configs specify corpus_type (which Flair corpus class to use) and corpus_args (folder location, column mappings, etc.)

### Model Configuration
- Model names point to HuggingFace Hub identifiers (e.g., `deepset/gbert-base`)
- Special handling for models that don't support QLoRA (flag: `supports_qlora`)
- Special handling for LLM2Vec bidirectional models (requires model merging via peft_paths)
- Some models have custom patches in `src/lib_patches/` for compatibility

### Library Patches
Located in `src/lib_patches/`, these patches handle library version mismatches and missing features:
- **flair_patches/**: Transformer embedding forward pass compatibility, dummy classifier
- **transformers_patches/**: Custom QA head implementations for specific models (Gemma2, EuroLLM)
- **sentence_transformer_patches/**: Transformer encoder modifications

These patches are imported and applied at module load time in the training scripts.

### Utility Functions (`src/utils.py`)
- `get_max_seq_length()`: Extracts max sequence length from model config with fallback to 512
- `get_peft_config()`: Builds LoRA configuration with custom target module overrides
- `get_bnb_config()`: Creates bitsandbytes quantization config with torch dtype conversion
- `create_weight_dict()`: Computes class weights from imbalanced training data

### Training Entry Point (`src/train.py`)
The main entry point uses Hydra to load configs and dispatches to appropriate training function based on `task.framework` field. It also:
- Handles LLM2Vec model merging before training
- Sets seeds for reproducibility
- Creates marker files to detect parallel run conflicts
- Handles data folder path resolution

### Output Structure
- Hydra creates timestamped output directories: `outputs/{date}/{time}/`
- Models and logs are saved within these directories under task-specific subdirectories
- Configuration used for each run is saved in `hydra_conf/` subdirectory

## Data Organization

The `data/` directory contains 10+ benchmark datasets, each with subdirectories for specific tasks:
- `Germeval/2025/`: GermEval 2025 shared task data (newest additions)
- Individual datasets: Argument_Mining, GermanQuAD, NER, PAWSX, Verbal_Idioms, etc.

Each dataset's README describes its format and task type.

## Adding New Tasks

To add a new task:
1. Create a new YAML file in `src/conf/task/` with task_name, framework, corpus_type, and corpus_args
2. Ensure the data is available at the path specified in corpus_args relative to `data/`
3. The corresponding Flair corpus class must exist or custom corpus loading must be added
4. Test with: `python src/train.py +task=your_new_task +model=gbert_base +train_args=a100`

## Adding New Models

To add a new model:
1. Create a new YAML file in `src/conf/model/` with model_name (HuggingFace ID) and supports_qlora flag
2. Optionally add peft_target_modules if the model needs custom LoRA targets
3. Test with: `python src/train.py +model=your_new_model +task=news_class +train_args=a100`

If the model requires custom patches (e.g., missing QA head), add to `src/lib_patches/` and import in the relevant training script.

## Key Dependencies

Main libraries:
- **hydra-core**: Configuration management
- **transformers**: Base model loading and utilities
- **flair**: Sequence tagging, text classification, embeddings
- **sentence-transformers**: Sentence similarity training
- **peft**: LoRA and quantization support
- **bitsandbytes**: 4-bit quantization
- **loguru**: Structured logging
- **datasets**: HuggingFace datasets API
