import random
from pathlib import Path
from unittest.mock import patch

import bitsandbytes as bnb
import hydra
import numpy as np
import torch
import transformers
from hydra.utils import get_original_cwd
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from transformers.models.auto.configuration_auto import AutoConfig

from train_flair import training as train_flair
from train_hf_qa import training as train_hf_qa
from train_st import training as train_st
from utils import from_config_QLORA_decorator, from_pretrained_QLORA_decorator


# https://hydra.cc/docs/configure_hydra/workdir/
@hydra.main(config_path="conf", config_name="config")
@patch("builtins.input", return_value="y")  # in order to accept custom code within HF models
def training(cfg: DictConfig, _magic_mock_trash=None):
    model_conf = AutoConfig.from_pretrained(cfg["model"]["model_name"])

    if "hf" in cfg["task"]["framework"] or cfg["train_args"]["disable_qlora"]:

        def patched_train(cfg: DictConfig = cfg):
            inner_train(cfg)

    elif (
        model_conf.model_type == "bert"  # no quantization for bert, as it's not supported
        or model_conf.model_type == "roberta"
        or "mt5" in model_conf.architectures[0].lower()  # don't get why this is necessary for mt5?
        or "gpt" in model_conf.architectures[0].lower()  # don't get why this is necessary for gpt?
    ):

        def patched_train(cfg: DictConfig = cfg):
            automodel_patched_train(cfg)

    else:

        @patch("torch.nn.Linear", bnb.nn.Linear8bitLt)
        @patch(
            "transformers.PreTrainedModel.to", lambda *args, **kwargs: args[0]
        )  # this might break literally everything
        def patched_train(cfg: DictConfig = cfg):
            automodel_patched_train(cfg)

    @patch(
        "transformers.AutoModel.from_config",
        from_config_QLORA_decorator(original_from_config=transformers.AutoModel.from_config, cfg=cfg),
    )
    @patch(
        "transformers.AutoModel.from_pretrained",
        from_pretrained_QLORA_decorator(original_from_pretrained=transformers.AutoModel.from_pretrained, cfg=cfg),
    )
    @patch(
        "transformers.T5EncoderModel.from_pretrained",
        from_pretrained_QLORA_decorator(original_from_pretrained=transformers.T5EncoderModel.from_pretrained, cfg=cfg),
    )
    def automodel_patched_train(cfg: DictConfig = cfg):
        inner_train(cfg)

    def inner_train(cfg: DictConfig = cfg, _magic_mock_trash=None):
        try:
            cfg["task"]["corpus_args"]["data_folder"] = cfg["data_base_dir"] + cfg["task"]["corpus_args"]["data_folder"]
            cfg["task"]["corpus_args"]["data_folder"] = str(
                (Path(get_original_cwd()) / cfg["task"]["corpus_args"]["data_folder"]).absolute()
            )
        except:
            pass

        logger.info(OmegaConf.to_yaml(cfg))

        random.seed(cfg["seed"])
        np.random.seed(cfg["seed"])
        torch.manual_seed(cfg["seed"])
        torch.cuda.manual_seed(cfg["seed"])
        torch.cuda.manual_seed_all(cfg["seed"])

        if cfg["task"]["framework"] == "flair":
            logger.info("Training with flair")
            train_flair(cfg)
        elif cfg["task"]["framework"] == "hf_qa":
            logger.info("Training with hf_qa")
            train_hf_qa(cfg)
        elif cfg["task"]["framework"] == "sentence_transformer":
            logger.info("Training with sentence_transformer")
            train_st(cfg)
        else:
            raise NotImplementedError("Not Implemented yet")

    patched_train(cfg)


if __name__ == "__main__":
    training()
