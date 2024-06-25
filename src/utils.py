import numpy as np
import pandas as pd
import torch
from flair.data import Corpus, Sentence, Token
from loguru import logger
from omegaconf import DictConfig
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training


def get_max_seq_length(cfg: DictConfig) -> int:
    from transformers.modeling_utils import PretrainedConfig
    from transformers.models.auto.configuration_auto import AutoConfig

    model_conf: PretrainedConfig = AutoConfig.from_pretrained(cfg["model"]["model_name"])

    possible_attrs = ["max_position_embeddings", "n_positions", "seq_length"]
    possible_attrs = [getattr(model_conf, attr, None) for attr in possible_attrs]
    possible_attrs = [attr for attr in possible_attrs if attr is not None] + [512]

    selected_attr = min(possible_attrs)
    logger.info(f"Using {selected_attr} as max_seq_length")

    return selected_attr


peft_config = LoraConfig(
    task_type="???",
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
)

bnb_config = {
    "device_map": "auto",
    "load_in_4bit": True,
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_quant_type": "nf4",
    "llm_int8_has_fp16_weight": True,
    "bnb_4bit_compute_dtype": torch.bfloat16,
}


def get_LORA_model(model, cfg: DictConfig):
    from transformers.models.auto.configuration_auto import AutoConfig

    if peft_config.task_type == "???":
        model_conf = AutoConfig.from_pretrained(cfg["model"]["model_name"])
        if cfg["task"]["framework"] in ["sentence_transformer", "flair"]:
            peft_config.task_type = TaskType.FEATURE_EXTRACTION
        elif cfg["task"]["framework"] == "hf_qa":
            peft_config.task_type = TaskType.QUESTION_ANS

    if "mistral" in model_conf.architectures[0].lower() or "mbart" in model_conf.architectures[0].lower():
        peft_config.target_modules = ["q_proj", "v_proj"]

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


def from_pretrained_QLORA_decorator(original_from_pretrained, cfg: DictConfig):
    if "hf" in cfg["task"]["framework"]:
        return original_from_pretrained
    # elif cfg["task"]["framework"] == "flair":
    #    if "mbart" in cfg["model"]["model_name"]:
    #        bnb_config["max_memory"] = {0: "20GB"}
    from transformers.models.auto.configuration_auto import AutoConfig

    model_conf = AutoConfig.from_pretrained(cfg["model"]["model_name"])

    def new_from_pretrained(args, **kwargs):
        if not "bert" in model_conf.architectures[0].lower():  # no quantization for bert, as it's not supported
            kwargs |= bnb_config
        model = original_from_pretrained(args, **kwargs)
        model = get_LORA_model(model, cfg=cfg)
        return model

    return new_from_pretrained


def from_config_QLORA_decorator(original_from_config, cfg: DictConfig):
    if "hf" in cfg["task"]["framework"]:
        return original_from_config
    # elif cfg["task"]["framework"] == "flair":
    #    if "mbart" in cfg["model"]["model_name"]:
    #        bnb_config["max_memory"] = {0: "20GB"}
    from transformers.models.auto.configuration_auto import AutoConfig

    model_conf = AutoConfig.from_pretrained(cfg["model"]["model_name"])

    def new_from_config(args, **kwargs):
        if not "bert" in model_conf.architectures[0].lower():  # no quantization for bert, as it's not supported
            kwargs["model_args"] = bnb_config
        model = original_from_config(args, **kwargs)
        model = get_LORA_model(model, cfg=cfg)
        return model

    return new_from_config


def create_weight_dict(data):
    df = pd.DataFrame([{"text": d.text, "label": [l.value for l in d.get_labels()]} for d in data])

    labels = np.array([label for label_list in df["label"] for label in label_list])
    unique_labels, counts = np.unique(labels, return_counts=True)
    class_weights = len(labels) / (len(unique_labels) * counts.astype(np.float64))
    class_weights_dict = dict(zip(unique_labels, class_weights))

    return class_weights_dict
