import random
from collections import Counter
from pathlib import Path
import pandas as pd
import numpy as np
from datasets import DatasetDict, load_from_disk
from loguru import logger
from omegaconf import DictConfig
from peft import TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForQuestionAnswering
from transformers.models.auto.modeling_auto import AutoModelForQuestionAnswering
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.mistral.configuration_mistral import MistralConfig
from transformers.models.mt5.configuration_mt5 import MT5Config
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.trainer import Trainer, TrainingArguments

from lib_patches.transformers_patches.LLaMaForQuestionAnswering import (
    LlamaForQuestionAnswering,
)
from lib_patches.transformers_patches.MistralForQuestionAnswering import (
    MistralForQuestionAnswering,
)
from lib_patches.transformers_patches.mt5ForQuestionsAnswering import (
    MT5ForQuestionAnswering,
)
from utils import bnb_config, get_max_seq_length, peft_config


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
    return prediction == ground_truth


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

    logger.add(Path.cwd() / cfg.task.task_name / "training_logs" / "logfile.log", level="INFO")

    # copied from here: https://huggingface.co/docs/transformers/tasks/question_answering
    logger.info("loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("loading and tokenizing dataset")

    qa_ds = load_from_disk(cfg["task"]["corpus_args"]["data_folder"])

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
        output_dir=Path.cwd() / cfg["task"]["task_name"] / "training_logs",
        learning_rate=cfg["train_args"]["learning_rate"],
        per_device_train_batch_size=cfg["train_args"]["batch_size"],
        per_device_eval_batch_size=cfg["train_args"]["batch_size"],
        num_train_epochs=1 if cfg["debug"] else cfg["train_args"]["epochs"],
        # fp16=cfg["train_args"]["precision"] == "fp16",
        gradient_accumulation_steps=cfg["train_args"]["gradient_accumulation_steps"]
        if cfg["train_args"]["gradient_accumulation_steps"]
        else 1,
        optim="paged_adamw_8bit",
        save_strategy="no",
    )

    logger.info("creating model")
    MT5ForQuestionAnswering.register_for_auto_class("AutoModelForQuestionAnswering")
    LlamaForQuestionAnswering.register_for_auto_class("AutoModelForQuestionAnswering")
    MistralForQuestionAnswering.register_for_auto_class("AutoModelForQuestionAnswering")

#    AutoModelForQuestionAnswering.register(LlamaConfig, LlamaForQuestionAnswering)
    AutoModelForQuestionAnswering.register(MistralConfig, MistralForQuestionAnswering)

    model = AutoModelForQuestionAnswering.from_pretrained(
        cfg["model"]["model_name"],
        **({} if "bert" in cfg["model"]["model_name"].lower() else bnb_config),
    )
    peft_config.task_type = TaskType.QUESTION_ANS

    if "mistral" in cfg["model"]["model_name"] or "mbart" in cfg["model"]["model_name"]:
        peft_config.target_modules = ["q_proj", "v_proj"]

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )

    model = get_peft_model(model, peft_config)

    # gpt2 has a weird token embedding size that is off by one
    # for all models where this already fits it's a noop
    if "gpt" in cfg["model"]["model_name"].lower():
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
        new_df['pred_label'] = flat_preds
        new_df['true_label'] = flat_labels
        path = Path.cwd() / cfg["task"]["task_name"] / "training_logs"
        new_df.to_csv(f'{path}/results.csv', header=True)
        f1 = f1_score(flat_preds, flat_labels)
        acc = exact_match_score(flat_preds, flat_labels)
        acc = sum(acc) / len(acc)
        return {
            "f1_score": f1,
            "acc_score": acc,
        }

    logger.info("creating trainer")

    if cfg["debug"]:
        if cfg["task"]["task_name"] == "mlqa":
            indices_to_keep_train = random.sample(range(len(tokenized_qa_ds["validation"])), 100)
            indices_to_keep_dev = random.sample(range(len(tokenized_qa_ds["validation"])), 100)
            indices_to_keep_test = random.sample(range(len(tokenized_qa_ds["test"])), 100)
            train_dataset = tokenized_qa_ds["validation"].select(indices_to_keep_train)
            dev_dataset = tokenized_qa_ds["validation"].select(indices_to_keep_dev)
            test_dataset = tokenized_qa_ds["test"].select(indices_to_keep_test)
        else:
            indices_to_keep_train = random.sample(range(len(tokenized_qa_ds["train"])), 100)
            indices_to_keep_test = random.sample(range(len(tokenized_qa_ds["test"])), 100)
            train_dataset = tokenized_qa_ds["train"].select(indices_to_keep_train)
            test_dataset = tokenized_qa_ds["test"].select(indices_to_keep_test)
    else:
        if cfg["task"]["task_name"] == "mlqa":
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
        compute_metrics=compute_metrics,
    )
    logger.info("starting training")
    trainer.train()

    logger.info("evaluating")

    eval_results = trainer.evaluate()
    logger.info(f"eval results: {eval_results}")

