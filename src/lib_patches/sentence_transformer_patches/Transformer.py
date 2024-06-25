from transformers import AutoModel, MT5Config, T5Config


# copied from git as this is already on master
def _load_model(self, model_name_or_path, config, cache_dir, **model_args):
    """Loads the transformer model"""
    if isinstance(config, T5Config):
        self._load_t5_model(model_name_or_path, config, cache_dir, **model_args)
    elif isinstance(config, MT5Config):
        self._load_mt5_model(model_name_or_path, config, cache_dir, **model_args)
    else:
        self.auto_model = AutoModel.from_pretrained(
            model_name_or_path, config=config, cache_dir=cache_dir, **model_args
        )


def _load_mt5_model(self, model_name_or_path, config, cache_dir, **model_args):
    """Loads the encoder model from T5"""
    from transformers import MT5EncoderModel

    MT5EncoderModel._keys_to_ignore_on_load_unexpected = ["decoder.*"]
    self.auto_model = MT5EncoderModel.from_pretrained(
        model_name_or_path, config=config, cache_dir=cache_dir, **model_args
    )
