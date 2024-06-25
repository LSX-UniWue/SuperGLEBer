import random
from pathlib import Path

import bitsandbytes as bnb
from datasets import DatasetDict, load_from_disk
from loguru import logger
from omegaconf import DictConfig
from sentence_transformers import InputExample, SentenceTransformer, losses, models
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader
import csv
import numpy as np
from scipy.stats import pearsonr
from scipy.spatial.distance import cosine
import sys
from lib_patches.sentence_transformer_patches import Transformer as TransformerPatches


Transformer = models.Transformer
Transformer._load_model = TransformerPatches._load_model
Transformer._load_mt5_model = TransformerPatches._load_mt5_model


def save_predictions(model: SentenceTransformer, examples: list[InputExample], output_file: str, batch_size: int = 16) -> None:
    sentences1 = [example.texts[0] for example in examples]
    sentences2 = [example.texts[1] for example in examples]
    true_labels = [example.label for example in examples]

    embeddings1 = model.encode(sentences1, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
    embeddings2 = model.encode(sentences2, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)

    cosine_scores = 1 - np.array([cosine(e1, e2) for e1, e2 in zip(embeddings1, embeddings2)])

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["sentence1", "sentence2", "true_label", "predicted_score"])
        for sent1, sent2, true_label, pred_score in zip(sentences1, sentences2, true_labels, cosine_scores):
            writer.writerow([sent1, sent2, true_label, pred_score])

    return true_labels, cosine_scores


def calculate_evaluation_score(true_labels, predicted_scores):
    pearson_corr, _ = pearsonr(true_labels, predicted_scores)
    return pearson_corr


def training(cfg: DictConfig) -> None:
    log_path = Path.cwd() / cfg.task.task_name / "training_logs"
    log_file = log_path / "logfile.log"
    logger.add(log_file, level="INFO", format="{time} {level} {message}", backtrace=True, diagnose=True)
    logger.add(sys.stdout, level="INFO", format="{time} {level} {message}")
    logger.add(sys.stderr, level="ERROR", format="{time} {level} {message}")


    # adapted from here: https://www.sbert.net/docs/training/overview.html and https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/sts/training_stsbenchmark.py
    logger.info("building model")
    word_embedding_model = Transformer(cfg["model"]["model_name"])
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

    if (
        "model_args" in cfg["model"]
        and "set_pad_token_to_eos_token_id" in cfg["model"]["model_args"]
        and cfg["model"]["model_args"]["set_pad_token_to_eos_token_id"]
    ):
        logger.info("patching tokenizer")
        model.tokenizer.pad_token = model.tokenizer.eos_token

    logger.info("loading dataset")
    hf_corpus = load_from_disk(cfg["task"]["corpus_args"]["data_folder"])

    try:
        similarity_score = "similarity_score"
        NORMALIZING_CONSTANT = max(hf_corpus["train"][similarity_score])
    except:
        similarity_score = "label"
        NORMALIZING_CONSTANT = max(hf_corpus["train"][similarity_score])

    sentence_transformer_corpus = dict()
    sentence_transformer_corpus_negative = {"train": [], "dev": [], "test": []}
    train_data = {}

    def add_to_samples(sent1, sent2, label):
        if sent1 not in train_data:
            train_data[sent1] = {"similar": set(), "dissimilar": set()}
        train_data[sent1][label].add(sent2)

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
                add_to_samples(sent1, sent2, "similar")
                add_to_samples(sent2, sent1, "similar")
            else:
                add_to_samples(sent1, sent2, "dissimilar")
                add_to_samples(sent2, sent1, "dissimilar")

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

    if cfg["debug"]:
        sentence_transformer_corpus_negative["train"] = sentence_transformer_corpus_negative["train"][:100]

    train_dataloader = DataLoader(
        sentence_transformer_corpus_negative["train"],
        shuffle=True,
        batch_size=cfg["train_args"]["batch_size"],
    )
    # train_loss = losses.CosineSimilarityLoss(model=model)
    train_loss = losses.MultipleNegativesRankingLoss(model=model)
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        examples=sentence_transformer_corpus["dev"],
        batch_size=cfg["train_args"]["batch_size"],
        name="sts-dev",
    )

    save_path = Path.cwd() / cfg["task"]["task_name"] / "training_logs"
    save_path.mkdir(parents=True, exist_ok=True)
    save_path = str(save_path)

    logger.info("starting training")

    model.fit(
        epochs=1 if cfg["debug"] else cfg["train_args"]["epochs"],
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
        batch_size=cfg["train_args"]["batch_size"],
        name="sts-test",
    )
    test_evaluator(model, output_path=save_path)

    logger.info("Saving predictions")
    predictions_output_file = Path(save_path) / "predictions.csv"
    true_labels, predicted_scores = save_predictions(model, sentence_transformer_corpus["test"], str(predictions_output_file))

    logger.info(f"Predictions saved to {predictions_output_file}")

    pearson_corr = calculate_evaluation_score(true_labels, predicted_scores)
    logger.info(f"Test set evaluation - Cosine-Similarity Pearson: {pearson_corr:.4f}")
