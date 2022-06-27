import logging

import torch
from transformers import PreTrainedTokenizerBase

from ..arguments import DatasetArguments
from .dataset_manager import DatasetManagerBase
from .external.utils_qa import prepare_train_features, prepare_validation_features
from .sampler import QAPossibleSubsetRandomSampler


logger = logging.getLogger(__name__)


class QADatasetManager(DatasetManagerBase):
    """
    Base dataset manager for SQuAD-like extractive QA tasks.
    """

    task_type = "question_answering"
    label_column_names = ["start_positions", "end_positions"]
    tasks_with_negatives = [
        "squad_v2",
        "duorc_p",
        "duorc_s",
    ]
    context_column_name = "context"
    question_column_name = "question"
    answer_column_name = "answers"

    def __init__(self, args: DatasetArguments, tokenizer: PreTrainedTokenizerBase = None):
        self.with_negative = (args.task_name or args.dataset_name) in self.tasks_with_negatives
        super().__init__(args, tokenizer=tokenizer, load_metric="squad_v2" if self.with_negative else "squad")
        self._encode_remove_columns = True

    def train_sampler(self):
        if (
            self.with_negative
            or self.args.train_subset_size < 0
            or self.args.train_subset_size >= len(self.train_split)
        ):
            return super().train_sampler()
        else:
            g = torch.Generator()
            if self.args.train_sampling_seed:
                g.manual_seed(self.args.train_sampling_seed)
            # only sample from the possible questions of the dataset
            return QAPossibleSubsetRandomSampler(
                self.train_split.features,
                self.args.train_subset_size,
                generator=g,
            )

    def encode_batch(self, examples):
        return prepare_train_features(
            examples,
            self.tokenizer,
            self.args.max_seq_length,
            self.args,
            self.context_column_name,
            self.question_column_name,
            self.answer_column_name,
        )

    def encode_batch_eval(self, examples):
        return prepare_validation_features(
            examples,
            self.tokenizer,
            self.args.max_seq_length,
            self.args,
            self.context_column_name,
            self.question_column_name,
            self.answer_column_name,
        )

    def get_prediction_head_config(self):
        return {
            "head_type": "question_answering",
            "num_labels": 2,
            "layers": 1,
            "activation_function": "tanh",
        }
