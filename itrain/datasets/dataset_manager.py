import inspect
import logging
import os
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from enum import IntEnum
from typing import List, Union

import numpy as np
from datasets import GenerateMode, Metric, Split, load_dataset, load_metric
from transformers import PreTrainedTokenizerBase, default_data_collator
from transformers.file_utils import torch_cache_home

from ..arguments import DatasetArguments
from ..ext.super_glue import SuperGlue


DATASET_FEATURES_CACHE = os.path.join(torch_cache_home, "itrain")
if not os.path.exists(DATASET_FEATURES_CACHE):
    os.makedirs(DATASET_FEATURES_CACHE)

logger = logging.getLogger(__name__)


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
    input_label_column_names = ["labels"]

    def __init__(
        self,
        args: DatasetArguments,
        tokenizer: PreTrainedTokenizerBase = None,
    ):
        self.args = args
        self._dataset_loc = self.args.dataset_dir or self.args.dataset_name
        self.tokenizer = tokenizer
        self.dataset = {}
        self.metric = None

    @property
    def name(self):
        if self.args.task_name:
            return f"{self.args.dataset_name}_{self.args.task_name}"
        else:
            return self.args.dataset_name

    def _get_features_cache_file(self, split_name):
        return os.path.join(
            DATASET_FEATURES_CACHE,
            "_".join([self.tokenizer.__class__.__name__, self.args.identifier, split_name]) + ".cached",
        )

    @abstractmethod
    def load(self, cache_mode: CacheMode = CacheMode.USE_DATASET_USE_FEATURES):
        pass

    def collate_fn(self, features):
        return default_data_collator(features)

    @abstractmethod
    def compute_metrics(self, predictions, references):
        pass

    @property
    def train_split(self):
        return self.dataset[Split.TRAIN] if Split.TRAIN in self.dataset else None

    @property
    def dev_split(self):
        return self.dataset[Split.VALIDATION] if Split.VALIDATION in self.dataset else None

    @property
    def test_split(self):
        return self.dataset[Split.TEST] if Split.TEST in self.dataset else None

    @abstractmethod
    def get_prediction_head_config(self):
        pass


class DatasetManagerBase(DatasetManager):

    def __init__(
        self,
        args: DatasetArguments,
        tokenizer: PreTrainedTokenizerBase = None,
        load_metric: Union[bool, Metric] = True,
    ):
        super().__init__(args, tokenizer)
        self._padding = "max_length"
        self._truncation = True
        self.load_metric = load_metric

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
            cache_files[split_name] = self._get_features_cache_file(split_name)
        self.dataset = self.dataset.map(
            self.encode_batch, batched=True, cache_file_names=cache_files, load_from_cache_file=load_from_cache
        )
        self.dataset.set_format(columns=["input_ids"] + self.label_column_names + self.tokenizer.model_input_names)

    def collate_fn(self, features):
        return default_data_collator(features)

    def compute_metrics(self, predictions, references):
        return self.metric.compute(predictions=predictions, references=references)


class GlueManager(DatasetManagerBase):
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
            truncation=self._truncation,
            padding=self._padding,
        )
        encoded.update({"labels": examples[self.column_config.label]})
        return encoded

    def compute_metrics(self, predictions, references):
        predictions = np.argmax(predictions, axis=1)
        return self.metric.compute(predictions=predictions, references=references)

    def get_prediction_head_config(self):
        return {
            "head_type": "classification",
            "num_labels": self.tasks_num_labels[self.args.task_name],
            "layers": 2,
            "activation_function": "tanh",
        }


class SuperGlueManager(DatasetManagerBase):
    tasks_num_labels = {
        "boolq": 2,
        "cb": 3,
        "copa": 2,
        "multirc": 2,
        "record": 2,
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
        elif self.args.task_name == "record":
            self.column_config = ColumnConfig(["passage", "query", "entities"], "answers")
            self._encode_batch = self._encode_batch_record
        # TODO wic ? wsc ?
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
            truncation=self._truncation,
            padding=self._padding,
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
            truncation=self._truncation,
            padding=self._padding,
        )
        encoded.update({"labels": examples[self.column_config.label]})
        return encoded

    def _encode_batch_record(self, examples):
        encoded = defaultdict(list)
        for idx, passage, query, entities, answers in zip(
            examples["idx"], examples["passage"], examples["query"], examples["entities"], examples["answers"]
        ):
            for entity in entities:
                label = 1 if entity in answers else 0
                query = query.replace("@placeholder", entity)
                example_encoded = self.tokenizer(
                    passage,
                    query,
                    max_length=self.args.max_seq_length,
                    truncation=True,
                    padding=self._padding,
                    return_overflowing_tokens=True,
                )
                if "overflowing_tokens" in example_encoded and len(example_encoded["overflowing_tokens"]) > 0:
                    logger.info("Cropping {0} tokens of input.".format(len(example_encoded["overflowing_tokens"])))
                encoded["idx"].append(idx)
                encoded["passage"].append(passage)
                encoded["query"].append(query)
                encoded["entities"].append(entity)
                encoded["answers"].append(answers)
                encoded["input_ids"].append(example_encoded["input_ids"])
                encoded["labels"].append(label)
                if "token_type_ids" in example_encoded:
                    encoded["token_type_ids"].append(example_encoded["token_type_ids"])
                if "attention_mask" in example_encoded:
                    encoded["attention_mask"].append(example_encoded["attention_mask"])
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
            truncation=self._truncation,
            padding=self._padding,
        )
        encoded.update({"labels": examples[self.column_config.label]})
        return encoded

    def compute_metrics(self, predictions, references):
        if self.args.task_name == "multirc":
            predictions = np.argmax(predictions, axis=1)
            predictions = [{"idx": idx, "prediction": pred} for idx, pred in zip(self.dev_split["idx"], predictions)]
        elif self.args.task_name == "record":
            max_preds = {}  # group predictions by question id
            for idx, entity, pred, answers in zip(
                self.dev_split["idx"], self.dev_split["entities"], predictions, self.dev_split["answers"]
            ):
                idx_string = f"{idx['passage']}-{idx['query']}"
                if idx_string not in max_preds or pred[1] > max_preds[idx_string]["logit"]:
                    max_preds[idx_string] = {"idx": idx, "logit": pred[1], "entity": entity, "answers": answers}
            predictions = [{"idx": val["idx"], "prediction_text": val["entity"]} for _, val in max_preds.items()]
            references = [{"idx": val["idx"], "answers": val["answers"]} for _, val in max_preds.items()]
        else:
            predictions = np.argmax(predictions, axis=1)
        return self.metric.compute(predictions=predictions, references=references)

    def get_prediction_head_config(self):
        return {
            "head_type": "classification",
            "num_labels": self.tasks_num_labels[self.args.task_name],
            "layers": 2,
            "activation_function": "tanh",
        }


class MultipleChoiceDatasetManager(DatasetManagerBase):
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
                truncation=self._truncation,
                padding=self._padding,
                return_overflowing_tokens=True,
            )
            if "overflowing_tokens" in encoded and len(encoded["overflowing_tokens"][0]) > 0:
                logger.info("Cropping {0} tokens of input.".format(len(encoded["overflowing_tokens"][0])))
            input_ids.append(encoded["input_ids"])
            if "token_type_ids" in encoded:
                token_type_ids.append(encoded["token_type_ids"])
            if "attention_mask" in encoded:
                attention_mask.append(encoded["attention_mask"])
        encoded = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": [
                self.choice_label_map[label] if label else None for label in examples[self.column_config.label]
            ],
        }
        if len(token_type_ids) > 0:
            encoded["token_type_ids"] = token_type_ids
        return encoded

    def compute_metrics(self, predictions, references):
        predictions = np.argmax(predictions, axis=1)
        return {"accuracy": (predictions == references).mean()}

    def get_prediction_head_config(self):
        return {
            "head_type": "multiple_choice",
            "num_choices": 4,
            "layers": 2,
            "activation_function": "tanh",
        }


class HellaswagManager(MultipleChoiceDatasetManager):
    def __init__(self, args: DatasetArguments, tokenizer: PreTrainedTokenizerBase = None):
        super().__init__(args, tokenizer)
        self.column_config = ColumnConfig(["ctx_a", "ctx_b", "endings"], "label")
        self.choice_label_map = {v: k for (k, v) in enumerate(["0", "1", "2", "3"])}


class RaceManager(MultipleChoiceDatasetManager):
    def __init__(self, args: DatasetArguments, tokenizer: PreTrainedTokenizerBase = None):
        super().__init__(args, tokenizer)
        self.column_config = ColumnConfig(["article", "question", "options"], "answer")
        self.choice_label_map = {v: k for (k, v) in enumerate(["A", "B", "C", "D"])}
