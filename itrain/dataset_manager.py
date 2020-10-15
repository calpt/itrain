import inspect
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from typing import List, Union

import numpy as np
import torch
from datasets import GenerateMode, Metric, load_dataset, load_metric
from transformers import PreTrainedTokenizerBase, default_data_collator
from transformers.file_utils import torch_cache_home

from .arguments import DatasetArguments
from .ext.super_glue import SuperGlue


DATASET_FEATURES_CACHE = os.path.join(torch_cache_home, "itrain")
if not os.path.exists(DATASET_FEATURES_CACHE):
    os.makedirs(DATASET_FEATURES_CACHE)


class CacheMode(IntEnum):
    NEW_DATASET_NEW_FEATURES = 0
    USE_DATASET_NEW_FEATURES = 1
    USE_DATASET_USE_FEATURES = 2


@dataclass
class ColumnConfig:
    inputs: List[str]
    label: str


class DatasetManager(ABC):
    label_column_names = ["labels"]

    def __init__(
        self,
        args: DatasetArguments,
        tokenizer: PreTrainedTokenizerBase = None,
        load_metric: Union[bool, Metric] = True,
    ):
        self.args = args
        self._dataset_loc = self.args.dataset_dir or self.args.dataset_name
        self.tokenizer = tokenizer
        self.load_metric = load_metric

    @property
    def name(self):
        if self.args.task_name:
            return f"{self.args.dataset_name}_{self.args.task_name}"
        else:
            return self.args.dataset_name

    def load(self, cache_mode: CacheMode = CacheMode.USE_DATASET_USE_FEATURES):
        # load dataset
        download_mode = (
            GenerateMode.FORCE_REDOWNLOAD
            if cache_mode == CacheMode.NEW_DATASET_NEW_FEATURES
            else GenerateMode.REUSE_DATASET_IF_EXISTS
        )
        self.dataset = load_dataset(
            self._dataset_loc,
            self.args.task_name,
            download_mode=download_mode,
        )
        # convert examples to transformers features
        self._encode(load_from_cache=cache_mode >= CacheMode.USE_DATASET_USE_FEATURES)
        # load metric
        if self.load_metric:
            if inspect.isclass(self.load_metric) and issubclass(self.load_metric, Metric):
                self.metric = self.load_metric(self.args.task_name)
            else:
                self.metric = load_metric(self.args.dataset_name, self.args.task_name)

    @abstractmethod
    def encode_batch(self, examples):
        pass

    def _encode(self, load_from_cache=True):
        cache_files = {}
        for split_name in self.dataset.keys():
            cache_files[split_name] = os.path.join(
                DATASET_FEATURES_CACHE,
                "_".join([self.tokenizer.__class__.__name__, self.args.identifier, split_name]) + ".arrow",
            )
        self.dataset = self.dataset.map(
            self.encode_batch, batched=True, cache_file_names=cache_files, load_from_cache_file=load_from_cache
        )
        self.dataset.set_format(columns=["input_ids"] + self.label_column_names + self.tokenizer.model_input_names)

    def collate_fn(self, features):
        return default_data_collator(features)

    def compute_metric(self, predictions, references):
        return self.metric.compute(predictions=predictions, references=references)

    @property
    def train_split(self):
        return self.dataset[self.split_names[0]] if self.split_names[0] else None

    @property
    def dev_split(self):
        return self.dataset[self.split_names[1]] if self.split_names[1] else None

    @property
    def test_split(self):
        return self.dataset[self.split_names[2]] if self.split_names[2] else None

    @abstractmethod
    def get_prediction_head_config(self):
        pass


class GlueManager(DatasetManager):
    dataset_id = "glue"
    split_names = ("train", "validation", "test")
    tasks_num_labels = {
        "cola": 2,
        "mnli": 3,
        "mrpc": 2,
        "sst-2": 2,
        "sts-b": 1,
        "qqp": 2,
        "qnli": 2,
        "rte": 2,
        "wnli": 2,
    }

    def __init__(self, args: DatasetArguments, tokenizer: PreTrainedTokenizerBase = None):
        super().__init__(args, tokenizer)
        self.column_config = self._get_column_config()

    def _get_column_config(self):
        if self.args.task_name == "mrpc" or self.args.task_name == "rte" or self.args.task_name == "wnli":
            return ColumnConfig(["sentence1", "sentence2"], "label")
        elif self.args.task_name == "sst2":
            return ColumnConfig(["sentence"], "label")
        elif self.args.task_name == "cola":
            return ColumnConfig(["sentence"], "is_acceptable")
        elif self.args.task_name == "qqp":
            return ColumnConfig(["question1", "question2"], "is_duplicate")
        elif self.args.task_name == "stsb":
            return ColumnConfig(["sentence1", "sentence2"], "score")
        elif self.args.task_name == "mnli":
            return ColumnConfig(["premise", "hypothesis"], "gold_label")
        elif self.args.task_name == "qnli":
            return ColumnConfig(["question", "sentence"], "label")
        else:
            raise ValueError()

    def encode_batch(self, examples):
        encoded = self.tokenizer(
            examples[self.column_config.inputs[0]],
            examples[self.column_config.inputs[1]] if len(self.column_config.inputs) > 1 else None,
            max_length=self.args.max_seq_length,
            truncation=True,
            padding="max_length",
        )
        encoded.update({"labels": examples[self.column_config.label]})
        return encoded

    def compute_metric(self, predictions, references):
        predictions = np.argmax(predictions, axis=1)
        return self.metric.compute(predictions=predictions, references=references)

    def get_prediction_head_config(self):
        return {
            "head_type": "classification",
            "num_labels": self.tasks_num_labels[self.args.task_name],
            "layers": 2,
            "activation_function": "tanh",
        }


class SuperGlueManager(DatasetManager):
    dataset_id = "super_glue"
    split_names = ("train", "validation", "test")
    tasks_num_labels = {
        "boolq": 2,
        "cb": 3,
        "copa": 2,
        "multirc": 2,
        # "record",
        "rte": 2,
        "wic": 2,
        "wsc": 2,
        "wsc.fixed": 2,
        "axb": 2,
        "axg": 2,
    }
    _COPA_DICT = {
        "cause": "What was the cause of this?",
        "effect": "What happened as a result?",
    }

    def __init__(self, args: DatasetArguments, tokenizer: PreTrainedTokenizerBase = None):
        super().__init__(args, tokenizer, load_metric=SuperGlue)
        if self.args.task_name == "boolq":
            self.column_config = ColumnConfig(["passage", "question"], "label")
            self._encode_batch = self._encode_batch_classification
        elif self.args.task_name == "cb" or self.args.task_name == "rte" or self.args.task_name == "axg":
            self.column_config = ColumnConfig(["premise", "hypothesis"], "label")
            self._encode_batch = self._encode_batch_classification
        elif self.args.task_name == "copa":
            self.column_config = ColumnConfig(["premise", "question", "choice1", "choice2"], "label")
            self._encode_batch = self._encode_batch_copa
        elif self.args.task_name == "multirc":
            self.column_config = ColumnConfig(["paragraph", "question", "answer"], "label")
            self._encode_batch = self._encode_batch_multirc
        elif self.args.task_name == "axb":
            self.column_config = ColumnConfig(["sentence1", "sentence2"], "label")
            self._encode_batch = self._encode_batch_classification
        # TODO record ? wic ? wsc ?
        else:
            raise ValueError()

    def encode_batch(self, examples):
        return self._encode_batch(examples)

    def _encode_batch_copa(self, examples):
        contexts = [p + " " + self._COPA_DICT[q] for p, q in zip(examples["premise"], examples["question"])]
        sentences_a = [ctx + " " + choice for ctx, choice in zip(contexts, examples["choice1"])]
        sentences_b = [ctx + " " + choice for ctx, choice in zip(contexts, examples["choice2"])]
        encoded = self.tokenizer(
            sentences_a,
            sentences_b,
            max_length=self.args.max_seq_length,
            truncation=True,
            padding="max_length",
        )
        encoded.update({"labels": examples[self.column_config.label]})
        return encoded

    def _encode_batch_multirc(self, examples):
        contexts = [
            paragraph + " " + question for paragraph, question in zip(examples["paragraph"], examples["question"])
        ]
        encoded = self.tokenizer(
            contexts,
            examples["answer"],
            max_length=self.args.max_seq_length,
            truncation=True,
            padding="max_length",
        )
        encoded.update({"labels": examples[self.column_config.label]})
        return encoded

    def _encode_batch_wic(self, examples):
        raise NotImplementedError()  # TODO WIC & WSC seem to need span classification ?

    def _encode_batch_wsc(self, examples):
        raise NotImplementedError()  # TODO WIC & WSC seem to need span classification ?

    def _encode_batch_classification(self, examples):
        encoded = self.tokenizer(
            examples[self.column_config.inputs[0]],
            examples[self.column_config.inputs[1]] if len(self.column_config.inputs) > 1 else None,
            max_length=self.args.max_seq_length,
            truncation=True,
            padding="max_length",
        )
        encoded.update({"labels": examples[self.column_config.label]})
        return encoded

    def compute_metric(self, predictions, references):
        predictions = np.argmax(predictions, axis=1)
        return self.metric.compute(predictions=predictions, references=references)

    def get_prediction_head_config(self):
        return {
            "head_type": "classification",
            "num_labels": self.tasks_num_labels[self.args.task_name],
            "layers": 2,
            "activation_function": "tanh",
        }


class SquadV1Manager(DatasetManager):
    dataset_id = "squad"
    split_names = ("train", "validation", None)
    label_column_names = ["start_positions", "end_positions"]

    def __init__(self, args: DatasetArguments, tokenizer: PreTrainedTokenizerBase = None):
        super().__init__(args, tokenizer)
        self.column_config = ColumnConfig(["context", "question"], "answers")

    def _get_correct_alignement(self, context, answer):
        """ Some original examples in SQuAD have indices wrong by 1 or 2 character. We test and fix this here. """
        # TODO
        if len(answer["text"]) < 1:
            return 0, 1
        gold_text = answer["text"][0]
        start_idx = answer["answer_start"][0]
        end_idx = start_idx + len(gold_text)
        if context[start_idx:end_idx] == gold_text:
            return start_idx, end_idx  # When the gold label position is good
        elif context[start_idx - 1 : end_idx - 1] == gold_text:
            return start_idx - 1, end_idx - 1  # When the gold label is off by one character
        elif context[start_idx - 2 : end_idx - 2] == gold_text:
            return start_idx - 2, end_idx - 2  # When the gold label is off by two character
        else:
            raise ValueError()

    def encode_batch(self, examples):
        encoded = self.tokenizer(
            examples["context"],
            examples["question"],
            max_length=self.args.max_seq_length,
            truncation=True,
            padding="max_length",
        )
        start_positions, end_positions = [], []
        for i, (context, answer) in enumerate(zip(examples["context"], examples["answers"])):
            start_idx, end_idx = self._get_correct_alignement(context, answer)
            assert encoded.char_to_token(i, start_idx) is not None
            start_positions.append(encoded.char_to_token(i, start_idx))
            end_positions.append(encoded.char_to_token(i, end_idx - 1))
        encoded.update({'start_positions': start_positions, 'end_positions': end_positions})
        return encoded

    def collate_fn(self, features):
        batch = default_data_collator(features)
        # HACK: fixes labels for adapter-transformers qa head
        batch["labels"] = torch.stack([batch["start_positions"], batch["end_positions"]])
        del batch["start_positions"]
        del batch["end_positions"]
        return batch

    def get_prediction_head_config(self):
        return {
            "head_type": "question_answering",
            "num_labels": 2,
            "layers": 1,
            "activation_function": "tanh",
        }


class SquadV2Manager(SquadV1Manager):
    dataset_id = "squad_v2"


class MultipleChoiceDatasetManager(DatasetManager):
    split_names = ("train", "validation", "test")

    def __init__(self, args: DatasetArguments, tokenizer: PreTrainedTokenizerBase = None):
        super().__init__(args, tokenizer, load_metric=False)

    def _build_input_choice(self, question, ending):
        if "_" in question:
            return question.replace("_", ending)
        else:
            return question + " " + ending

    def encode_batch(self, examples):
        input_ids = []
        token_type_ids = []
        attention_mask = []
        for context, question, endings in zip(*[examples[c] for c in self.column_config.inputs]):
            a_s = [context for _ in range(4)]
            b_s = [self._build_input_choice(question, endings[i]) for i in range(4)]
            encoded = self.tokenizer(
                a_s,
                b_s,
                max_length=self.args.max_seq_length,
                truncation=True,
                padding="max_length",
            )
            input_ids.append(encoded["input_ids"])
            if "token_type_ids" in encoded:
                token_type_ids.append(encoded["token_type_ids"])
            if "attention_mask" in encoded:
                attention_mask.append(encoded["attention_mask"])
        encoded = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "labels": [
                self.choice_label_map[label] if label else None for label in examples[self.column_config.label]
            ],
        }
        return encoded

    def compute_metric(self, predictions, references):
        predictions = np.argmax(predictions, axis=1)
        return {"acc": (predictions == references).mean()}

    def get_prediction_head_config(self):
        return {
            "head_type": "multiple_choice",
            "num_choices": 4,
            "layers": 2,
            "activation_function": "tanh",
        }


class HellaswagManager(MultipleChoiceDatasetManager):
    dataset_id = "hellaswag"

    def __init__(self, args: DatasetArguments, tokenizer: PreTrainedTokenizerBase = None):
        super().__init__(args, tokenizer)
        self.column_config = ColumnConfig(["ctx_a", "ctx_b", "endings"], "label")
        self.choice_label_map = {v: k for (k, v) in enumerate(["1", "2", "3", "4"])}


class RaceManager(MultipleChoiceDatasetManager):
    dataset_id = "race"

    def __init__(self, args: DatasetArguments, tokenizer: PreTrainedTokenizerBase = None):
        super().__init__(args, tokenizer)
        self.column_config = ColumnConfig(["article", "question", "options"], "answer")
        self.choice_label_map = {v: k for (k, v) in enumerate(["A", "B", "C", "D"])}


DATASET_MANAGER_CLASSES = {}
for name, obj in globals().copy().items():
    if inspect.isclass(obj) and issubclass(obj, DatasetManager) and hasattr(obj, "dataset_id"):
        DATASET_MANAGER_CLASSES[obj.dataset_id] = obj
