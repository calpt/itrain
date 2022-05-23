from typing import Union

import numpy as np
import torch
from datasets import ClassLabel, Metric, concatenate_datasets
from transformers import PreTrainedTokenizerBase

from ..arguments import DatasetArguments
from .dataset_manager import CacheMode, ColumnConfig, DatasetManagerBase
from .sampler import StratifiedRandomSampler


class ClassificationDatasetManager(DatasetManagerBase):
    """
    Base dataset manager for sequence classification tasks.
    """

    tasks_num_labels = {
        "imdb": 2,
        "rotten_tomatoes": 2,
        "emo": 4,
        "emotion": 6,
        "yelp_polarity": 2,
        "scicite": 3,
        "snli": 3,
        "trec": 6,
        "eraser_multi_rc": 2,
        "sick": 3,
        "ag_news": 4,
    }

    def __init__(
        self,
        args: DatasetArguments,
        tokenizer: PreTrainedTokenizerBase = None,
        load_metric: Union[bool, Metric] = False,
    ):
        super().__init__(args, tokenizer, load_metric=load_metric)
        self._no_stratification = False
        self._configure()

    def train_sampler(self):
        if (
            self.args.train_subset_size <= 0
            or self.args.train_subset_size >= len(self.train_split)
            or self._no_stratification
        ):
            return super().train_sampler()
        else:
            g = torch.Generator()
            if self.args.train_sampling_seed:
                g.manual_seed(self.args.train_sampling_seed)
            return StratifiedRandomSampler(
                self.train_split[self.column_config.label],
                self.args.train_subset_size,
                generator=g,
            )

    def _configure(self):
        if self.args.dataset_name == "scicite":
            self.column_config = ColumnConfig(["string"], "label")
        elif self.args.dataset_name == "trec":
            self.column_config = ColumnConfig(["text"], "label-coarse")
        elif self.args.dataset_name == "snli":
            self.column_config = ColumnConfig(["premise", "hypothesis"], "label")
        elif self.args.dataset_name == "eraser_multi_rc":
            self.column_config = ColumnConfig(["passage", "query_and_answer"], "label")
        elif self.args.dataset_name == "sick":
            self.column_config = ColumnConfig(["sentence_A", "sentence_B"], "label")
        else:
            self.column_config = ColumnConfig(["text"], "label")

    def load(self, cache_mode: CacheMode = CacheMode.USE_DATASET_USE_FEATURES):
        super().load(cache_mode=cache_mode)
        if isinstance(self.train_split.features[self.column_config.label], ClassLabel):
            self.label_list = self.train_split.features[self.column_config.label].names
        else:
            labels = self.train_split[self.column_config.label] + self.dev_split[self.column_config.label]
            if self.test_split:
                labels += self.test_split[self.column_config.label]
            if not isinstance(labels[0], float):
                self.label_list = self._get_label_list(labels)
            else:
                self.label_list = None

    def _map_labels(self, examples):
        return examples[self.column_config.label]

    def _custom_filter(self, example):
        return example[self.column_config.label] > -1

    def encode_batch(self, examples):
        encoded = self.tokenizer(
            examples[self.column_config.inputs[0]],
            examples[self.column_config.inputs[1]] if len(self.column_config.inputs) > 1 else None,
            max_length=self.args.max_seq_length,
            truncation=self._truncation,
            padding=self._padding,
        )
        encoded.update({"labels": self._map_labels(examples)})
        return encoded

    def compute_metrics(self, predictions, references):
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        predictions = np.argmax(predictions, axis=1)
        return {"accuracy": (predictions == references).mean()}

    @property
    def num_labels(self):
        return self.tasks_num_labels[self.args.task_name or self.args.dataset_name]

    def get_prediction_head_config(self):
        return {
            "head_type": "classification",
            "num_labels": self.num_labels,
            "layers": 2,
            "activation_function": "tanh",
            "label2id": {v: k for k, v in enumerate(self.label_list)} if self.label_list else None,
        }


class WikiQAManager(ClassificationDatasetManager):
    tasks_num_labels = {
        "wiki_qa": 2,
    }

    def _configure(self):
        self.column_config = ColumnConfig(["question", "answer"], "label")


class SciTailManager(ClassificationDatasetManager):
    tasks_num_labels = {
        "scitail": 2,
    }
    label_map = {
        "neutral": 0,
        "entails": 1,
    }

    def _map_labels(self, examples):
        return [self.label_map.get(label, None) for label in examples[self.column_config.label]]

    def _custom_filter(self, example):
        return example[self.column_config.label] in self.label_map

    def _configure(self):
        self._use_task_name_for_loading = False
        self._default_subset_name = "tsv_format"
        self.column_config = ColumnConfig(["premise", "hypothesis"], "label")


class ANLIManager(ClassificationDatasetManager):
    tasks_num_labels = {
        "r1": 3,
        "r2": 3,
        "r3": 3,
    }

    def _configure(self):
        self._use_task_name_for_loading = False
        self.column_config = ColumnConfig(["premise", "hypothesis"], "label")
        self.train_split_name = "train_" + self.args.task_name
        self.dev_split_name = "dev_" + self.args.task_name
        self.test_split_name = "test_" + self.args.task_name

    @property
    def train_split(self):
        if self.args.task_name == "r1":
            return self.dataset["train_r1"]
        elif self.args.task_name == "r2":
            return concatenate_datasets([self.dataset["train_r1"], self.dataset["train_r2"]])
        elif self.args.task_name == "r3":
            return concatenate_datasets([self.dataset["train_r1"], self.dataset["train_r2"], self.dataset["train_r3"]])
        else:
            raise ValueError("Invalid task_name")
