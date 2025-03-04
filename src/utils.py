import os.path

import numpy as np
import pandas as pd
import torch
from loguru import logger
from omegaconf import DictConfig
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig


def get_max_seq_length(cfg: DictConfig) -> int:
    if cfg["model"]["model_name"] == "dummy":
        return 512
    from transformers.modeling_utils import PretrainedConfig
    from transformers.models.auto.configuration_auto import AutoConfig

    model_conf: PretrainedConfig = AutoConfig.from_pretrained(cfg["model"]["model_name"])

    possible_attrs = ["max_position_embeddings", "n_positions", "seq_length"]
    possible_attrs = [getattr(model_conf, attr, None) for attr in possible_attrs]
    possible_attrs = [attr for attr in possible_attrs if attr is not None] + [512]

    selected_attr = min(possible_attrs)
    logger.info(f"Using {selected_attr} as max_seq_length")

    return selected_attr


def get_peft_config(cfg: DictConfig) -> LoraConfig:
    peft_config = LoraConfig(**cfg.train_procedure.peft_config)
    if "peft_target_modules" in cfg.model:
        peft_config.target_modules = cfg.model.peft_target_modules
    if peft_config.task_type == "???":
        from peft.utils.peft_types import TaskType

        peft_config.task_type = TaskType.FEATURE_EXTRACTION
    return peft_config


def get_bnb_config(cfg: DictConfig) -> BitsAndBytesConfig:
    bnb_config = BitsAndBytesConfig(**cfg.train_procedure.bnb_config)
    if not cfg.model.get("supports_qlora", True):
        bnb_config = {}
    if "bnb_4bit_compute_dtype" in bnb_config and type(bnb_config["bnb_4bit_compute_dtype"]) == str:
        bnb_config["bnb_4bit_compute_dtype"] = getattr(torch, bnb_config["bnb_4bit_compute_dtype"])
    return bnb_config


def create_weight_dict(data):
    df = pd.DataFrame([{"text": d.text, "label": [l.value for l in d.get_labels()]} for d in data])

    labels = np.array([label for label_list in df["label"] for label in label_list])
    unique_labels, counts = np.unique(labels, return_counts=True)
    class_weights = len(labels) / (len(unique_labels) * counts.astype(np.float64))
    class_weights_dict = dict(zip(unique_labels, class_weights))

    return class_weights_dict


def merge_llm2vec(model_path, peft_path, out_dir):
    """util function to merge loras obtained from llm2vec"""

    from llm2vec import LLM2Vec

    model = LLM2Vec.from_pretrained(
        model_path,
        peft_model_name_or_path=peft_path,
        merge_peft=True,
        enable_bidirectional=True,
        device_map="cpu",
        torch_dtype=torch.bfloat16,
    )

    model.save(out_dir, merge_before_save=True)

def merge_multiple_llm2vec(cfg):
    """basically a loop through `merge_llm2vec` to merge multiple adapters"""

    peft_paths = cfg['model'].get("peft_paths", None)
    suffix = cfg['model'].get("merged_suffix", None)
    new_model_path = str(cfg.model.model_name) + (suffix if suffix else "_merged")

    for i, lora_path in enumerate(peft_paths):
        if i == 0:
            merge_llm2vec(
                model_path=cfg.model.model_name,
                peft_path=lora_path,
                out_dir=new_model_path,
            )
        else:
            merge_llm2vec(
                model_path=new_model_path,
                peft_path=lora_path,
                out_dir=new_model_path,
            )

    cfg.model.model_name = new_model_path
    cfg.model.peft_paths = None


def patch_transformers_automodel():
    """util function to overwrite the AutoModel classes to their respective llm2vec classes (bidirectional models from causal models)"""

    from transformers import AutoModel
    from transformers import (
        LlamaConfig,
        Qwen2Config,
        GemmaConfig,
        MistralConfig
    )
    from llm2vec.models import (
        LlamaBiModel,
        Qwen2BiModel,
        GemmaBiModel,
        MistralBiModel,
    )

    # only supporting the models used in the llm2vec repo itself
    for config, bimodel in zip(
        [LlamaConfig, Qwen2Config, GemmaConfig, MistralConfig],
        [LlamaBiModel, Qwen2BiModel, GemmaBiModel, MistralBiModel],
    ):
        bimodel.register_for_auto_class("AutoModel")
        AutoModel.register(config, bimodel, exist_ok=True)


def patch_transformers_automodelforqna():
    """same as `patch_transformers_automodel` but on the AutoModelForQuestionAnswering classes"""

    from transformers import AutoModelForQuestionAnswering
    from transformers import LlamaConfig
    from lib_patches.transformers_patches.BiLlamaForQuestionAnswering import BiLlamaForQuestionAnswering

    # TODO: only supporting llama for now --> write qna wrappers for the others
    for config, bimodel in zip(
        [LlamaConfig],
        [BiLlamaForQuestionAnswering],
    ):
        bimodel.register_for_auto_class("AutoModelForQuestionAnswering")
        AutoModelForQuestionAnswering.register(config, bimodel, exist_ok=True)


def create_config_for_llm2vec(base_model_dir, config_path, base_adapter_dir, mntp_steps, simcse_steps=None, custom_simcse=None):
    """assumes a given directory structure to create respective configs"""

    from jinja2 import Environment, FileSystemLoader

    merged_suffix = ""
    peft_paths = []

    if mntp_steps not in [1_000, 10_000]:
        raise ValueError(f"SimCSE has only trained on mntp models with either 1k or 10k steps, got {mntp_steps} instead")

    mntp_path = os.path.join(base_adapter_dir, f"mntp/llammlein7b-mntp/checkpoint-{mntp_steps}")
    merged_suffix += f"_mntp-{mntp_steps}"
    peft_paths.append(mntp_path)

    simcse_exists = simcse_steps is not None and custom_simcse is not None
    if simcse_exists:
        simcse_path = os.path.join(base_adapter_dir, "simcse")
        from_mntp_steps = "1k_to_simcse_fewer_steps" if mntp_steps == 1_000 else "10k_to_simcse"
        simcse_path = os.path.join(simcse_path, from_mntp_steps)
        merged_suffix += f"_simcse"

        if custom_simcse:
            simcse_path = os.path.join(simcse_path, "custom")
            merged_suffix += f"-custom"

        simcse_path = os.path.join(simcse_path, f"checkpoint-{simcse_steps}")
        merged_suffix += f"-{simcse_steps}"
        peft_paths.append(simcse_path)

    template_dict = {
        "base_model_dir": base_model_dir,
        "peft_paths": peft_paths,
        "suffix": merged_suffix,
    }

    template = Environment(loader=FileSystemLoader(config_path)).get_template("llm2vec_template.yaml")
    output_yaml = template.render(**template_dict)

    new_config_name = f"llm2vec_for{merged_suffix}.yaml"
    with open(os.path.join(config_path, new_config_name), "w") as f:
        f.write(output_yaml)

    print(new_config_name)
