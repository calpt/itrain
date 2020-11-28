import logging

import numpy as np
from transformers import PreTrainedTokenizerBase

from ..arguments import DatasetArguments
from .dataset_manager import ColumnConfig, DatasetManagerBase


logger = logging.getLogger(__name__)


class MultipleChoiceDatasetManager(DatasetManagerBase):
    def __init__(self, args: DatasetArguments, tokenizer: PreTrainedTokenizerBase = None):
        super().__init__(args, tokenizer, load_metric=False)

    def _custom_filter(self, example):
        return example[self.column_config.label] in self.choice_label_map and all(
            [example[col] is not None for col in self.column_config.inputs]
        )

    def _build_input_choice(self, question, ending):
        if "_" in question:
            return question.replace("_", ending)
        else:
            return question + " " + ending

    def _build_inputs(self, example):
        context, question, endings = example
        a_s = [context for _ in range(len(self.choice_label_map))]
        b_s = [self._build_input_choice(question, endings[i]) for i in range(len(self.choice_label_map))]
        return a_s, b_s

    def encode_batch(self, examples):
        input_ids = []
        token_type_ids = []
        attention_mask = []
        for example in zip(*[examples[c] for c in self.column_config.inputs]):
            a_s, b_s = self._build_inputs(example)
            encoded = self.tokenizer(
                a_s,
                b_s,
                max_length=self.args.max_seq_length,
                truncation=self._truncation,
                padding=self._padding,
                return_overflowing_tokens=True,
            )
            # if "overflowing_tokens" in encoded and len(encoded["overflowing_tokens"][0]) > 0:
            #     logger.info("Cropping {0} tokens of input.".format(len(encoded["overflowing_tokens"][0])))
            input_ids.append(encoded["input_ids"])
            if "token_type_ids" in encoded:
                token_type_ids.append(encoded["token_type_ids"])
            if "attention_mask" in encoded:
                attention_mask.append(encoded["attention_mask"])
        encoded = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": [self.choice_label_map.get(label, None) for label in examples[self.column_config.label]],
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
            "num_choices": len(self.choice_label_map),
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


class QuailManager(MultipleChoiceDatasetManager):
    def __init__(self, args: DatasetArguments, tokenizer: PreTrainedTokenizerBase = None):
        super().__init__(args, tokenizer)
        self.column_config = ColumnConfig(["context", "question", "answers"], "correct_answer_id")
        self.choice_label_map = {v: k for (k, v) in enumerate(range(4))}


class ARTManager(MultipleChoiceDatasetManager):
    def __init__(self, args: DatasetArguments, tokenizer: PreTrainedTokenizerBase = None):
        super().__init__(args, tokenizer)
        self.column_config = ColumnConfig(["observation_1", "observation_2", "hypothesis_1", "hypothesis_2"], "label")
        self.choice_label_map = {v: k for (k, v) in enumerate([1, 2])}

    def _build_inputs(self, example):
        a_s = [example[0] + " " + example[2], example[0] + " " + example[3]]
        b_s = [example[1] for _ in range(2)]
        return a_s, b_s
