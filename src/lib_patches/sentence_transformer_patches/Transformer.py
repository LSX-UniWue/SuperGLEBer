from omegaconf import DictConfig
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModel, MT5Config, T5Config

from utils import get_bnb_config, get_peft_config


def get_load_model(cfg: DictConfig):
    def _load_model(self, model_name_or_path, config, cache_dir, backend="torch", is_peft_model=False, *args,
                    **model_args):
        """Loads the transformer model"""

        if "bnb_config" in cfg.train_procedure:
            if config.model_type == "bert" or config.model_type == "modernbert":  # bert does not support quantization
                model_args |= {}
            else:
                model_args |= {"quantization_config": get_bnb_config(cfg)}

        if isinstance(config, T5Config):
            self._load_t5_model(model_name_or_path, config, cache_dir, **model_args)
        elif isinstance(config, MT5Config):
            self._load_mt5_model(model_name_or_path, config, cache_dir, **model_args)
        else:
            self.auto_model = AutoModel.from_pretrained(
                model_name_or_path, config=config, cache_dir=cache_dir, **model_args
            )

        if "peft_config" in cfg.train_procedure:
            if "bnb_config" in cfg.train_procedure:
                self.auto_model = prepare_model_for_kbit_training(
                    self.auto_model, gradient_checkpointing_kwargs={"use_reentrant": False}
                )

            peft_config = get_peft_config(cfg)
            self.auto_model = get_peft_model(self.auto_model, peft_config)

    return _load_model