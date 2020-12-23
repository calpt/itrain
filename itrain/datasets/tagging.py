import numpy as np
from datasets import ClassLabel
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import DataCollatorForTokenClassification, PreTrainedTokenizerBase

from ..arguments import DatasetArguments
from .dataset_manager import CacheMode, ColumnConfig, DatasetManagerBase


task_num_labels = {
    "conll2003": 9,
    "conll2000": 23,
    "universal_dependencies": 18,
}

class TaggingDatasetManager(DatasetManagerBase):
    """
    This class is adapted from the run_ner.py example script of HuggingFace transformers.
    """

    def __init__(
        self,
        args: DatasetArguments,
        tokenizer: PreTrainedTokenizerBase = None,
        column_config: ColumnConfig = None,
    ):
        super().__init__(args, tokenizer, load_metric=False)
        self.column_config = column_config or self._get_column_config()
        self.num_labels = task_num_labels[self.args.dataset_name]

    def _get_column_config(self):
        if self.args.dataset_name == "conll2003":
            return ColumnConfig("tokens", "ner_tags")
        elif self.args.dataset_name == "conll2000":
            return ColumnConfig("tokens", "chunk_tags")
        elif self.args.dataset_name == "universal_dependencies":
            return ColumnConfig("tokens", "upos")
        else:
            raise ValueError("No ColumnConfig specified.")

    def _get_label_list(labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list

    def load(self, cache_mode: CacheMode = CacheMode.USE_DATASET_USE_FEATURES):
        super().load(cache_mode=cache_mode)
        if isinstance(self.train_split.features[self.column_config.label].feature, ClassLabel):
            self.label_list = self.train_split.features[self.column_config.label].feature.names
            self.label_to_id = {i: i for i in range(len(self.label_list))}
        else:
            self.label_list = self._get_label_list(self.train_split.features[self.column_config.label])
            self.label_to_id = {l: i for i, l in enumerate(self.label_list)}
        assert self.num_labels >= len(self.label_list)
        self.collate_fn = DataCollatorForTokenClassification(self.tokenizer)

    def encode_batch(self, examples):
        tokenized_inputs = self.tokenizer(
            examples[self.column_config.inputs],
            max_length=self.args.max_seq_length,
            padding=self._padding,
            truncation=self._truncation,
            is_split_into_words=True,
        )
        labels = []
        for i, label in enumerate(examples[self.column_config.label]):
            word_ids = tokenized_inputs.words(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(self.label_to_id[label[word_idx]])
                # For the other tokens in a word, we set the label to -100
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx

            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def compute_metrics(self, predictions, references):
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [self.label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, references)
        ]
        true_labels = [
            [self.label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, references)
        ]

        return {
            "accuracy": accuracy_score(true_labels, true_predictions),
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
        }

    def get_prediction_head_config(self):
        return {
            "head_type": "tagging",
            "num_labels": self.num_labels,
            "layers": 1,
            "activation_function": "tanh",
        }
