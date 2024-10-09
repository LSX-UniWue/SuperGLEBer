import csv
import random
import sys
from pathlib import Path

import bitsandbytes as bnb
import numpy as np
import pandas as pd
from datasets import DatasetDict, Dataset
from loguru import logger
from omegaconf import DictConfig
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
from sentence_transformers import InputExample, SentenceTransformer, losses, models
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader

from lib_patches.sentence_transformer_patches import Transformer as TransformerPatches


def save_predictions(model: SentenceTransformer, examples: list[InputExample], output_file: str, batch_size: int = 16):
    sentences1 = [example.texts[0] for example in examples]
    sentences2 = [example.texts[1] for example in examples]
    true_labels = [example.label for example in examples]

    embeddings1 = model.encode(sentences1, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
    embeddings2 = model.encode(sentences2, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)

    cosine_scores = 1 - np.array([cosine(e1, e2) for e1, e2 in zip(embeddings1, embeddings2)])

    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["sentence1", "sentence2", "true_label", "predicted_score"])
        for sent1, sent2, true_label, pred_score in zip(sentences1, sentences2, true_labels, cosine_scores):
            writer.writerow([sent1, sent2, true_label, pred_score])

    return true_labels, cosine_scores


def calculate_evaluation_score(true_labels, predicted_scores):
    pearson_corr, _ = pearsonr(true_labels, predicted_scores)
    return pearson_corr


def training(cfg: DictConfig) -> None:

    Transformer = models.Transformer
    Transformer._load_model = TransformerPatches.get_load_model(cfg)

    log_path = Path.cwd() / cfg.task.task_name / "training_logs"
    log_file = log_path / "logfile.log"
    logger.add(log_file, level="INFO", format="{time} {level} {message}", backtrace=True, diagnose=True)
    logger.add(sys.stdout, level="INFO", format="{time} {level} {message}")
    logger.add(sys.stderr, level="ERROR", format="{time} {level} {message}")

    # adapted from here: https://www.sbert.net/docs/training/overview.html and https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/sts/training_stsbenchmark.py
    logger.info("building model")
    word_embedding_model = Transformer(cfg.model.model_name)
    # gpt2 has a weird token embedding size that is off by one
    # for all models where this already fits it's a noop
    word_embedding_model.auto_model.resize_token_embeddings(len(word_embedding_model.tokenizer))

    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False,
    )
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    if cfg.model.get("set_pad_token_to_eos_token_id", False):
        model.tokenizer.pad_token = model.tokenizer.eos_token

    logger.info("loading dataset")

    train_path = f'{cfg.task.corpus_args.data_folder}/train.tsv'
    test_path = f'{cfg.task.corpus_args.data_folder}/test.tsv'
    dev_path = f'{cfg.task.corpus_args.data_folder}/dev.tsv'

    train_df = pd.read_csv(train_path, delimiter='\t', quoting=3)
    test_df = pd.read_csv(test_path, delimiter='\t', quoting=3)
    dev_df = pd.read_csv(dev_path, delimiter='\t', quoting=3)

    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    dev_dataset = Dataset.from_pandas(dev_df)

    hf_corpus = DatasetDict({
        'train': train_dataset,
        'test': test_dataset,
        'validation': dev_dataset
    })

    print(hf_corpus['train'][0])


    try:
        similarity_score = "similarity_score"
        NORMALIZING_CONSTANT = max(hf_corpus["train"][similarity_score])
    except:
        similarity_score = "label"
        NORMALIZING_CONSTANT = max(hf_corpus["train"][similarity_score])

    sentence_transformer_corpus = dict()
    sentence_transformer_corpus_negative = {"train": [], "dev": [], "test": []}
    train_data = {}

    def add_to_samples(s1, s2, label):
        if s1 not in train_data:
            train_data[s1] = {"similar": set(), "dissimilar": set()}
        train_data[s1][label].add(s2)

    for split in hf_corpus.keys():
        # https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/nli/training_nli_v2.py#L74
        sentence_transformer_corpus["dev" if split == "validation" else split] = [
            InputExample(
                texts=[example["sentence1"], example["sentence2"]],
                label=example[similarity_score] / NORMALIZING_CONSTANT,
            )
            for example in hf_corpus[split]
        ] + [
            InputExample(
                texts=[example["sentence2"], example["sentence1"]],
                label=example[similarity_score] / NORMALIZING_CONSTANT,
            )
            for example in hf_corpus[split]
        ]
        threshold_similar = 0.5
        for row in hf_corpus[split]:
            sent1 = row["sentence1"].strip()
            sent2 = row["sentence2"].strip()
            score = float(row[similarity_score]) / NORMALIZING_CONSTANT

            if score >= threshold_similar:
                add_to_samples(s1=sent1, s2=sent2, label="similar")
                add_to_samples(sent2, sent1, label="similar")
            else:
                add_to_samples(s1=sent1, s2=sent2, label="dissimilar")
                add_to_samples(s1=sent2, s2=sent1, label="dissimilar")

        for sent1, others in train_data.items():
            if len(others["similar"]) > 0 and len(others["dissimilar"]) > 0:
                sentence_transformer_corpus_negative["dev" if split == "validation" else split].append(
                    InputExample(
                        texts=[sent1, random.choice(list(others["similar"])), random.choice(list(others["dissimilar"]))]
                    )
                )
                sentence_transformer_corpus_negative["dev" if split == "validation" else split].append(
                    InputExample(
                        texts=[random.choice(list(others["similar"])), sent1, random.choice(list(others["dissimilar"]))]
                    )
                )

    del hf_corpus

    train_dataloader = DataLoader(
        sentence_transformer_corpus_negative["train"],
        shuffle=True,
        batch_size=cfg.train_args.batch_size,
    )
    train_loss = losses.MultipleNegativesRankingLoss(model=model)
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        examples=sentence_transformer_corpus["dev"],
        batch_size=cfg.train_args.batch_size,
        name="sts-dev",
    )

    save_path = Path.cwd() / cfg.task.task_name / "training_logs"
    save_path.mkdir(parents=True, exist_ok=True)
    save_path = str(save_path)

    logger.info("starting training")

    model.fit(
        epochs=cfg.train_args.epochs,
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        evaluation_steps=1000,
        optimizer_class=bnb.optim.adamw.AdamW,
        output_path=save_path,
        save_best_model=False,
    )

    logger.info("evaluating on test set")
    test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        examples=sentence_transformer_corpus["test"],
        batch_size=cfg.train_args.batch_size,
        name="sts-test",
    )
    test_evaluator(model, output_path=save_path)

    logger.info("Saving predictions")
    predictions_output_file = Path(save_path) / "predictions.csv"
    true_labels, predicted_scores = save_predictions(
        model, sentence_transformer_corpus["test"], str(predictions_output_file)
    )

    logger.info(f"Predictions saved to {predictions_output_file}")

    pearson_corr = calculate_evaluation_score(true_labels, predicted_scores)
    logger.info(f"Test set evaluation - Cosine-Similarity Pearson: {pearson_corr:.4f}")
