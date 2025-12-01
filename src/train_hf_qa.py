from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from datasets import DatasetDict, load_from_disk
from loguru import logger
from omegaconf import DictConfig
from peft import TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoConfig
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.trainer import Trainer, TrainingArguments
from lib_patches.transformers_patches.Gemma2ForQuestionAnswering import Gemma2ForQuestionAnswering
from lib_patches.transformers_patches.EuroBERTForQuestionAnswering import EuroBertForQuestionAnswering
from transformers.models.gemma2.configuration_gemma2 import Gemma2Config
from lib_patches.transformers_patches.EuroBERTForQuestionAnswering import EuroBertConfig

from transformers import AutoModel, AutoModelForQuestionAnswering
import sys

from utils import get_bnb_config, get_max_seq_length, get_peft_config


def setup_logging(log_dir: Path):
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "train.log"
    logger.add(log_file)
    sys.stdout = open(log_file, "a")
    sys.stderr = open(log_file, "a")


def f1_score(prediction, ground_truth):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (prediction) == (ground_truth)


def training(cfg: DictConfig) -> None:
    def preprocess_function(examples, tokenizer: PreTrainedTokenizer, max_length: int = get_max_seq_length(cfg)):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=max_length,
            truncation=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            answer = answers[i]
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while idx < len(sequence_ids) and sequence_ids[idx] != 1:
                idx += 1
            context_start = idx

            while idx < len(sequence_ids) and sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label it (0, 0)
            if (
                    context_start >= len(offset)
                    or context_end >= len(offset)
                    or offset[context_start][0] > end_char
                    or offset[context_end][1] < start_char
            ):
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    log_dir = Path.cwd() / cfg.task.task_name / "training_logs"
    setup_logging(log_dir)

    # copied from here: https://huggingface.co/docs/transformers/tasks/question_answering
    logger.info("loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("loading and tokenizing dataset")
    qa_ds = load_from_disk(cfg.task.corpus_args.data_folder)

    logger.info(f"first sample: {qa_ds['test'][0]}")

    tokenized_qa_ds: DatasetDict = qa_ds.map(
        preprocess_function,
        batched=True,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_length": get_max_seq_length(cfg),
        },
    )

    training_args = TrainingArguments(
        output_dir=Path.cwd() / cfg.task.task_name / "training_logs",
        learning_rate=cfg.train_args.learning_rate,
        per_device_train_batch_size=cfg.train_args.batch_size,
        per_device_eval_batch_size=cfg.train_args.batch_size,
        num_train_epochs=cfg.train_args.epochs,
        seed=cfg.seed,
        fp16=cfg.train_args.get("fp16", False),
        gradient_accumulation_steps=(
            cfg.train_args.gradient_accumulation_steps if cfg.train_args.gradient_accumulation_steps else 1
        ),
        optim="paged_adamw_8bit",
        save_strategy="no",
    )

    logger.info("creating model")
    Gemma2ForQuestionAnswering.register_for_auto_class("AutoModelForQuestionAnswering")
    AutoModelForQuestionAnswering.register(Gemma2Config, Gemma2ForQuestionAnswering)
    AutoConfig.register("eurobert", EuroBertConfig)
    EuroBertForQuestionAnswering.register_for_auto_class("AutoModelForQuestionAnswering")
    AutoModelForQuestionAnswering.register(EuroBertConfig, EuroBertForQuestionAnswering)
    config = AutoConfig.from_pretrained(cfg.model.model_name, trust_remote_code=False,
                                        finetuning_task="question-answering",
                                        **cfg.model.get("model_config_args", {}))

    bnb_config = {}
    if "bnb_config" in cfg.train_procedure:
        if config.model_type == "bert" or config.model_type == "modernbert" or config.model_type == "eurobert":  # bert does not support quantization
            bnb_config = {}
        else:
            bnb_config = {"quantization_config": get_bnb_config(cfg)}


    # https://github.com/huggingface/transformers/issues/30381#issuecomment-2120004654 - weights are not initialized
    if  config.model_type == "llama" or config.model_type == "mistral":
        model = AutoModel.from_pretrained(cfg.model.model_name)
        model.save_pretrained("tmp")
        model = AutoModelForQuestionAnswering.from_pretrained("tmp",
                                                              config=config,
                                                              **bnb_config,
                                                              **cfg.model.get("model_args", {}),
                                                              )

    else:
        model = AutoModelForQuestionAnswering.from_pretrained(
            cfg.model.model_name,
            config=config,
            **bnb_config,
            **cfg.model.get("model_args", {}),
        )
    if "peft_config" in cfg.train_procedure:
        cfg.train_procedure.peft_config.task_type = TaskType.QUESTION_ANS
        peft_config = get_peft_config(cfg)

        if "bnb_config" in cfg.train_procedure:
            model = prepare_model_for_kbit_training(model, gradient_checkpointing_kwargs={"use_reentrant": False})
        model = get_peft_model(model, peft_config)

    # for all models where this already fits it's a noop
    if cfg.model.get("set_pad_token_to_eos_token_id", False):
        model.resize_token_embeddings(len(tokenizer))

    def compute_metrics(p):
        preds = p.predictions
        if len(preds) == 3:
            preds = preds[:2]
        label_ids = p.label_ids
        max_last_dim = np.argmax(preds, axis=2)
        preds_2d = max_last_dim.reshape((-1, max_last_dim.shape[-1]))
        flat_labels = np.array(label_ids).flatten()
        flat_preds = preds_2d.flatten()
        new_df = pd.DataFrame()
        new_df["pred_label"] = flat_preds
        new_df["true_label"] = flat_labels
        path = Path.cwd() / cfg["task"]["task_name"] / "training_logs"
        logger.info(f"writing results to {path}")
        new_df.to_csv(f"{path}/results.csv", header=True)

        f1 = f1_score(flat_preds, flat_labels)
        acc = exact_match_score(flat_preds, flat_labels)
        acc = sum(acc) / len(acc)
        logger.info(f"f1: {f1}, acc: {acc}")
        return {
            "f1_score": f1,
            "acc_score": acc,
        }

    logger.info("creating trainer")

    if cfg.task.task_name == "mlqa":
        train_dataset = tokenized_qa_ds["validation"]
    else:
        train_dataset = tokenized_qa_ds["train"]

    test_dataset = tokenized_qa_ds["test"]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,  # https://github.com/huggingface/peft/issues/1120
    )
    logger.info("starting training")
    trainer.train()

    logger.info("evaluating")

    eval_results = trainer.evaluate()
    logger.info(f"eval results: {eval_results}")
