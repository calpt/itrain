import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch


@dataclass
class DatasetArguments:

    dataset_name: str = field(metadata={"help": "Name of the dataset to be loaded."})
    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "Name of the task or configuration to be loaded."},
    )
    dataset_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to a local loading script to optionally load a local dataset."
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )

    @property
    def identifier(self):
        return "_".join([self.dataset_name, self.task_name or "", str(self.max_seq_length)])


@dataclass
class ModelArguments:

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={"help": "Specifies whether to use Hugginface's Fast Tokenizers."},
    )
    train_adapter: bool = field(
        default=False,
        metadata={"help": "Train an adapter instead of the full model."},
    )
    adapter_config: Optional[str] = field(
        default="pfeiffer",
        metadata={"help": "Adapter configuration. Either an identifier or a path to a file."},
    )


@dataclass
class RunArguments:

    output_dir: Optional[str] = field(
        default=None,
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )

    evaluate_during_training: bool = field(
        default=True,
        metadata={"help": "Run evaluation during training after each epoch."},
    )
    patience: int = field(
        default=0,
        metadata={
            "help": "If > 0 stops training after evaluating this many times consecutively with non-decreasing loss."
        },
    )
    patience_metric: str = field(default="eval_loss", metadata={"help": "Metric used for early stopping. Loss by default."})

    batch_size: int = field(default=16, metadata={"help": "Batch size."})

    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )

    learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate for Adam."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay if we apply some."})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for Adam optimizer."})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})

    num_train_epochs: float = field(default=3.0, metadata={"help": "Total number of training epochs to perform."})
    max_steps: int = field(
        default=-1,
        metadata={"help": "If > 0: set total number of training steps to perform. Override num_train_epochs."},
    )
    warmup_steps: int = field(default=0, metadata={"help": "Linear warmup over warmup_steps."})

    logging_dir: Optional[str] = field(default=None, metadata={"help": "Tensorboard log dir."})

    checkpoint_epochs: int = field(default=0, metadata={"help": "Save model checkpoint after every X epochs."})
    save_total_limit: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Limit the total amount of checkpoints."
                "Deletes the older checkpoints in the output_dir. Default is unlimited checkpoints"
            )
        },
    )

    seed: int = field(default=42, metadata={"help": "random seed for initialization"})

    past_index: int = field(
        default=-1,
        metadata={"help": "If >=0, uses the corresponding part of the output as the past state for next step."},
    )

    @property
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def to_json_string(self):
        """
        Serializes this instance to a JSON string.
        """
        return json.dumps(dataclasses.asdict(self), indent=2)

    def to_sanitized_dict(self) -> Dict[str, Any]:
        """
        Sanitized serialization to use with TensorBoard’s hparams
        """
        d = dataclasses.asdict(self)
        valid_types = [bool, int, float, str, torch.Tensor]
        return {k: v if type(v) in valid_types else str(v) for k, v in d.items()}
