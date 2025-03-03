import random
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.utils import get_original_cwd
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from transformers import set_seed

from train_flair import training as train_flair
from train_hf_qa import training as train_hf_qa
from train_st import training as train_st

from utils import merge_multiple_llm2vec


# https://hydra.cc/docs/configure_hydra/workdir/
@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def training(cfg: DictConfig):
    try:
        cfg.task.corpus_args.data_folder = cfg.data_base_dir + cfg.task.corpus_args.data_folder
        cfg.task.corpus_args.data_folder = str((Path(get_original_cwd()) / cfg.task.corpus_args.data_folder).absolute())
    except:
        pass

    # to verify that no two runs have the same directory
    Path(cfg.model.model_name.replace("/", "_") + ".model").touch()
    Path(cfg.task.task_name.replace("/", "_") + ".task").touch()

    logger.info(OmegaConf.to_yaml(cfg))

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    set_seed(cfg.seed)

    # enable correct model merging for llm2vec before starting training
    if cfg['model'].get("peft_paths", None) and cfg['model'].get("is_bidirectional", False):
        merge_multiple_llm2vec(cfg)

    if cfg.task.framework == "flair":
        logger.info("Training with flair")
        train_flair(cfg)
    elif cfg.task.framework == "hf_qa":
        logger.info("Training with hf_qa")
        train_hf_qa(cfg)
    elif cfg.task.framework == "sentence_transformer":
        logger.info("Training with sentence_transformer")
        train_st(cfg)

    else:
        raise NotImplementedError("tja")


if __name__ == "__main__":
    training()
