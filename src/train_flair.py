from pathlib import Path

import bitsandbytes as bnb
import flair
import torch
from flair.data import Corpus
from flair.datasets import (
    ClassificationCorpus,
    ColumnCorpus,
    CSVClassificationCorpus,
    DataPairCorpus,
    DataTripleCorpus,
)
from flair.embeddings import (
    TransformerDocumentEmbeddings,
    TransformerEmbeddings,
    TransformerWordEmbeddings,
)
from flair.models import (
    SequenceTagger,
    TextClassifier,
    TextPairClassifier,
    TextPairRegressor,
    TextTripleClassifier,
)
from flair.nn import Classifier
from flair.trainers import ModelTrainer
from loguru import logger
from omegaconf import DictConfig
from peft import LoraConfig
from transformers import AutoConfig

from lib_patches.flair_patches.dummyclassifier import DummyTextClassifier
from utils import (
    create_weight_dict,
    get_bnb_config,
    get_max_seq_length,
    get_peft_config,
)


def training(cfg: DictConfig) -> None:
    model_conf = AutoConfig.from_pretrained(cfg["model"]["model_name"])
    logger.add(Path.cwd() / cfg.task.task_name / "training_logs" / "logfile.log", level="INFO")
    if cfg.train_args.get("use_mps", False):
        flair.device = torch.device("mps")
    logger.info(flair.device)

    logger.info("loading corpus")
    corpus: Corpus = globals()[cfg.task.corpus_type](
        **cfg.task.get("corpus_args", {}),
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
    if cfg.task.classifier_type in ["TextClassifier", "TextPairClassifier", "TextTripleClassifier"]:
        additional_classifier_args["label_dictionary"] = corpus.make_label_dictionary(label_type=cfg.task.label_type)
    elif cfg.task.classifier_type == "SequenceTagger":
        additional_classifier_args["tag_dictionary"] = corpus.make_label_dictionary(
            label_type=cfg.task.label_type, add_unk=True
        )
        if "multi_label" in cfg.task:
            additional_classifier_args["multi_label"] = cfg.task.get("multi_label", False)

    logger.info(f"label distribution: {corpus.get_label_distribution()}")

    logger.info("creating model")
    weight_dict = create_weight_dict(data=corpus.train)
    logger.info(f"weight_dict has been created {weight_dict}")

    if cfg["model"]["model_name"] == "dummy":
        classifier: flair.nn.Classifier = DummyTextClassifier(
            baseline_type=cfg.model.baseline_type,
            classifier_type=cfg.task.classifier_type,
            label_type=cfg.task.label_type,
            multi_label=additional_classifier_args.get("multi_label", False),
        )
    else:
        if "peft_config" in cfg.train_procedure:
            peft_config = get_peft_config(cfg)
        else:
            peft_config = None

        bnb_config = {}
        if "bnb_config" in cfg.train_procedure:
            if (
                model_conf.model_type == "bert" or model_conf.model_type == "modernbert"
            ):  # bert does not support quantization
                bnb_config = {}
            else:
                bnb_config = {"quantization_config": get_bnb_config(cfg)}

        classifier: flair.nn.Classifier = globals()[cfg.task.classifier_type](
            embeddings=globals()[cfg.task.embedding_type](
                model=cfg.model.model_name,
                fine_tune=cfg.train_procedure.get("fine_tune", True),
                force_max_length=True,
                transformers_config_kwargs=cfg.model.get("model_config_args", {}),
                transformers_model_kwargs=bnb_config | dict(cfg.model.get("model_args", {})),
                transformers_tokenizer_kwargs={"model_max_length": get_max_seq_length(cfg)},
                peft_config=peft_config if "peft_config" in cfg.train_procedure else None,
                peft_gradient_checkpointing_kwargs={"gradient_checkpointing_kwargs": {"use_reentrant": False}},
            ),
            **cfg.task.get("classifier_args", {}),
            **additional_classifier_args,
            loss_weights=weight_dict,
        )

    if cfg.model.get("set_pad_token_to_eos_token_id", False):
        logger.info("patching tokenizer")
        transfo_embeddings: TransformerEmbeddings = classifier.embeddings
        transfo_embeddings.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        transfo_embeddings.model.resize_token_embeddings(len(transfo_embeddings.tokenizer))

    if cfg.model.get("set_padding_right", False):
        transfo_embeddings: TransformerEmbeddings = classifier.embeddings
        transfo_embeddings.tokenizer.padding_side = "right"

    logger.info("creating trainer")

    trainer = ModelTrainer(model=classifier, corpus=corpus)

    train_results = trainer.fine_tune(
        base_path=Path.cwd() / cfg.task.task_name / "training_logs",
        learning_rate=cfg.train_args.learning_rate,
        mini_batch_size=(
            cfg.train_args.batch_size * cfg.train_args.gradient_accumulation_steps
            if cfg.train_args.gradient_accumulation_steps
            else cfg.train_args.batch_size
        ),
        max_epochs=cfg.train_args.epochs,
        optimizer=bnb.optim.adamw.AdamW,
        use_amp=cfg.train_procedure.get("fp16", False),
        mini_batch_chunk_size=cfg.train_args.batch_size,
        save_final_model=False,
    )
    logger.info(f"train results: {train_results}")
