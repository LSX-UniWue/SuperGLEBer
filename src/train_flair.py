from pathlib import Path

import bitsandbytes as bnb
import flair
import torch
from flair.data import Corpus, Sentence, Token
from flair.datasets import (
    GERMEVAL_2018_OFFENSIVE_LANGUAGE,
    NEL_GERMAN_HIPE,
    NER_GERMAN_BIOFID,
    NER_GERMAN_EUROPARL,
    NER_GERMAN_GERMEVAL,
    NER_GERMAN_LEGAL,
    UD_GERMAN,
    UD_GERMAN_HDT,
    UP_GERMAN,
    ClassificationCorpus,
    ColumnCorpus,
    CSVClassificationCorpus,
    DataPairCorpus,
)
from flair.embeddings import TransformerDocumentEmbeddings, TransformerWordEmbeddings
from flair.models import SequenceTagger, TextClassifier, TextPairClassifier
from flair.nn import Classifier
from flair.trainers import ModelTrainer
from loguru import logger
from omegaconf import DictConfig

from lib_patches.flair_patches.triple import DataTripleCorpus, TextTripleClassifier
from utils import (
    create_weight_dict,
    get_max_seq_length,
)


def training(cfg: DictConfig) -> None:
    logger.add(Path.cwd() / cfg.task.task_name / "training_logs" / "logfile.log", level="INFO")
    if cfg["train_args"].get("use_mps", False):
        flair.device = torch.device("mps")
    logger.info(flair.device)

    logger.info("loading corpus")
    corpus: Corpus = globals()[cfg["task"]["corpus_type"]](
            **cfg["task"]["corpus_args"] or {},
    )

    logger.info(f"first sample: {corpus.get_all_sentences()[0]}")

    if cfg["debug"]:
        logger.info("Debug mode")
        percent = 100 / len(corpus.train)
        corpus.downsample(percentage=percent)
        logger.info(
            f"Debug-Corpus: {len(corpus.train)} train + {len(corpus.dev)} dev + {len(corpus.test)} test sentences"
        )

    corpus.filter_empty_sentences()

    additional_classifier_args = dict()
    if cfg["task"]["classifier_type"] in ["TextClassifier", "TextPairClassifier", "TextTripleClassifier"]:
        additional_classifier_args["label_dictionary"] = corpus.make_label_dictionary(
            label_type=cfg["task"]["label_type"]
        )
    elif cfg["task"]["classifier_type"] == "SequenceTagger":
        additional_classifier_args["tag_dictionary"] = corpus.make_label_dictionary(
            label_type=cfg["task"]["label_type"], add_unk=True
        )

    logger.info(f"label distribution: {corpus.get_label_distribution()}")

    logger.info("creating model")
    weight_dict = create_weight_dict(data=corpus.train)
    logger.info(f"weight_dict has been created {weight_dict}")

    if cfg["task"]["classifier_type"] == "SequenceTagger":
        classifier: flair.nn.Classifier = globals()[cfg["task"]["classifier_type"]](
            embeddings=globals()[cfg["task"]["embedding_type"]](
                model=cfg["model"]["model_name"],
                fine_tune=True,
                force_max_length=True,
                model_max_length=get_max_seq_length(cfg),
            ),
            **cfg["task"]["classifier_args"] or {},
            **additional_classifier_args,
            loss_weights=weight_dict,
        )
    else:
        classifier: flair.nn.Classifier = globals()[cfg["task"]["classifier_type"]](
            embeddings=globals()[cfg["task"]["embedding_type"]](
                model=cfg["model"]["model_name"],
                fine_tune=True,
                force_max_length=True,
                model_max_length=get_max_seq_length(cfg),
            ),
            **cfg["task"]["classifier_args"] or {},
            **additional_classifier_args,
            loss_weights=weight_dict,
            multi_label=True if "multi_label" in cfg["task"] else False,
        )

    if (
        "model_args" in cfg["model"]
        and "set_pad_token_to_eos_token_id" in cfg["model"]["model_args"]
        and cfg["model"]["model_args"]["set_pad_token_to_eos_token_id"]
    ):
        logger.info("patching tokenizer")
        classifier.embeddings.tokenizer.pad_token_id = classifier.embeddings.tokenizer.eos_token_id
        classifier.embeddings.tokenizer.pad_token = classifier.embeddings.tokenizer.eos_token

    logger.info("creating trainer")
    trainer = ModelTrainer(model=classifier, corpus=corpus)

    train_results = trainer.fine_tune(
        base_path=Path.cwd() / cfg["task"]["task_name"] / "training_logs",
        learning_rate=cfg["train_args"]["learning_rate"],
        mini_batch_size=cfg["train_args"]["batch_size"] * cfg["train_args"]["gradient_accumulation_steps"]
        if cfg["train_args"]["gradient_accumulation_steps"]
        else cfg["train_args"]["batch_size"],
        max_epochs=1 if cfg["debug"] else cfg["train_args"]["epochs"],
        optimizer=bnb.optim.adamw.AdamW,
        # use_amp=cfg["train_args"]["precision"] == "fp16",
        mini_batch_chunk_size=cfg["train_args"]["batch_size"],
        save_final_model=False,
    )
    logger.info(f"train results: {train_results}")
