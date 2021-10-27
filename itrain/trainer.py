import random
import time
from typing import Optional

import numpy as np
import torch
from transformers import AdapterTrainer as TransformersAdapterTrainer
from transformers import EarlyStoppingCallback, PreTrainedModel
from transformers import Trainer as TransformersTrainer

from .arguments import RunArguments
from .datasets import DatasetManager


def set_seed(seed: int):
    if seed is None:
        seed = int((time.time() * 1000) % 2 ** 32)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


class TrainerMixin:
    model: PreTrainedModel
    dataset_manager: DatasetManager

    def __init__(
        self,
        model: PreTrainedModel,
        args: RunArguments,
        dataset_manager: DatasetManager,
    ):
        super().__init__(
            model,
            args.to_hf_training_args(),
            data_collator=dataset_manager.collate_fn,
            train_dataset=dataset_manager.train_split,
            eval_dataset=dataset_manager.dev_split,
            tokenizer=dataset_manager.tokenizer,
            compute_metrics=lambda p: dataset_manager.compute_metrics(p.predictions, p.label_ids),
        )
        self.dataset_manager = dataset_manager
        self.label_names = self.dataset_manager.label_column_names

        # early stopping
        if args.patience > 0:
            callback = EarlyStoppingCallback(args.patience)
            self.add_callback(callback)

    # Override this method because AutoModelWithHeads wouldn't work otherwise.
    def _remove_unused_columns(self, dataset, description: Optional[str] = None):
        return dataset

    def _get_train_sampler(self) -> Optional[torch.utils.data.sampler.Sampler]:
        return self.dataset_manager.train_sampler()


class AdapterTrainer(TrainerMixin, TransformersAdapterTrainer):
    pass


class FineTuningTrainer(TrainerMixin, TransformersTrainer):
    pass
