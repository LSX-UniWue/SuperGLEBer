from dataclasses import dataclass
from typing import Optional


@dataclass
class TaskConfig:
    task_name: str
    framework: str
    multi_label: bool
    embedding_type: str
    classifier_type: str
    label_type: str
    classifier_args: dict[str, str]
    corpus_type: str
    corpus_args: dict[str, str]


@dataclass
class ModelConfig:
    model_name: Optional[str]
    baseline_type: Optional[str]
    set_pad_token_to_eos_token_id: bool
    peft_target_modules: Optional[list[str]]
    model_args: Optional[dict[str, str]]


@dataclass
class TrainArgs:
    learning_rate: float
    batch_size: int
    gradient_accumulation_steps: int
    epochs: int


@dataclass
class TrainProcedure:
    fine_tune: bool
    fp16: bool
    bnb_config: dict[str, str]
    peft_config: dict[str, str]


@dataclass
class SuperGLEBerConfig:
    data_base_dir: str
    seed: int
    debug: bool
    task: TaskConfig
    model: ModelConfig
    train_args: TrainArgs
    train_procedure: TrainProcedure
