from collections import defaultdict
from typing import Union

import numpy as np
from datasets import Metric, concatenate_datasets
from transformers import PreTrainedTokenizerBase

from ..arguments import DatasetArguments
from ..ext.super_glue import SuperGlue
from .dataset_manager import ColumnConfig, DatasetManagerBase
from .sampler import StratifiedRandomSampler


class ClassificationDatasetManager(DatasetManagerBase):
    tasks_num_labels = {
        "imdb": 2,
        "rotten_tomatoes": 2,
        "emo": 4,
        "yelp_polarity": 2,
        "scicite": 3,
        "snli": 3,
        "trec": 6,
        "eraser_multi_rc": 2,
        "sick": 3,
    }

    def __init__(
        self,
        args: DatasetArguments,
        tokenizer: PreTrainedTokenizerBase = None,
        load_metric: Union[bool, Metric] = False,
    ):
        super().__init__(args, tokenizer, load_metric=load_metric)
        self._configure()

    def train_sampler(self):
        if self.args.train_subset_size <= 0 or self.args.train_subset_size >= len(self.train_split):
            return super().train_sampler()
        else:
            return StratifiedRandomSampler(
                self.train_split[self.column_config.label],
                self.args.train_subset_size,
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


class EmotionDatasetManager(ClassificationDatasetManager):
    tasks_num_labels = {
        "emotion": 6,
    }

    emotion_label2id = {
        "sadness": 0,
        "joy": 1,
        "love": 2,
        "anger": 3,
        "fear": 4,
        "surprise": 5,
    }

    def _map_labels(self, examples):
        return [self.emotion_label2id[label] for label in examples[self.column_config.label]]

    def _custom_filter(self, example):
        return example[self.column_config.label] in self.emotion_label2id


class GlueManager(ClassificationDatasetManager):
    tasks_num_labels = {
        "cola": 2,
        "mnli": 3,
        "mrpc": 2,
        "sst2": 2,
        "stsb": 1,
        "qqp": 2,
        "qnli": 2,
        "rte": 2,
        "wnli": 2,
    }

    def __init__(self, args: DatasetArguments, tokenizer: PreTrainedTokenizerBase = None):
        super().__init__(args, tokenizer, load_metric=True)

    def _configure(self):
        if (
            self.args.task_name == "mrpc"
            or self.args.task_name == "rte"
            or self.args.task_name == "wnli"
            or self.args.task_name == "stsb"
        ):
            self.column_config = ColumnConfig(["sentence1", "sentence2"], "label")
        elif self.args.task_name == "sst2":
            self.column_config = ColumnConfig(["sentence"], "label")
        elif self.args.task_name == "cola":
            self.column_config = ColumnConfig(["sentence"], "label")
        elif self.args.task_name == "qqp":
            self.column_config = ColumnConfig(["question1", "question2"], "label")
        elif self.args.task_name == "mnli":
            self.column_config = ColumnConfig(["premise", "hypothesis"], "label")
            self.dev_split_name = "validation_matched"
            self.test_split_name = "test_matched"
        elif self.args.task_name == "qnli":
            self.column_config = ColumnConfig(["question", "sentence"], "label")
        else:
            raise ValueError()

    def compute_metrics(self, predictions, references):
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        if self.args.task_name == "stsb":
            predictions = np.squeeze(predictions)
        else:
            predictions = np.argmax(predictions, axis=1)
        return self.metric.compute(predictions=predictions, references=references)


class SuperGlueManager(ClassificationDatasetManager):
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

    def _configure(self):
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
        elif self.args.task_name == "wic":
            self.column_config = ColumnConfig(["sentence1", "sentence2", "word"], "label")
            self._encode_batch = self._encode_batch_wic
        # TODO wsc ?
        else:
            raise ValueError()

    def _custom_filter(self, example):
        return True  # Override for custom filtering

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
                query_filled = query.replace("@placeholder", entity)
                example_encoded = self.tokenizer(
                    passage,
                    query_filled,
                    max_length=self.args.max_seq_length,
                    truncation=self._truncation,
                    padding=self._padding,
                    return_overflowing_tokens=True,
                )
                # if "overflowing_tokens" in example_encoded and len(example_encoded["overflowing_tokens"]) > 0:
                #     logger.info("Cropping {0} tokens of input.".format(len(example_encoded["overflowing_tokens"])))
                encoded["idx"].append(idx)
                encoded["passage"].append(passage)
                encoded["query"].append(query_filled)
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
        sentences = []
        # concat all three input parts with [SEP] token
        for parts in zip(*[examples[c] for c in self.column_config.inputs]):
            sentences.append(self.tokenizer.sep_token.join(parts))
        encoded = self.tokenizer(
            sentences,
            max_length=self.args.max_seq_length,
            truncation=self._truncation,
            padding=self._padding,
        )
        encoded.update({"labels": examples[self.column_config.label]})
        return encoded

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
        if isinstance(predictions, tuple):
            predictions = predictions[0]
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
