"""
This module heavily relies on https://github.com/adapter-hub/adapter-transformers/blob/master/examples/pytorch/summarization/run_summarization.py.
"""
import logging

import nltk
import numpy as np
from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizerBase

from ..arguments import DatasetArguments
from .dataset_manager import DatasetManagerBase


logger = logging.getLogger(__name__)


summarization_name_mapping = {
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "cnn_dailymail": ("article", "highlights"),
    "orange_sum": ("text", "summary"),
    "pn_summary": ("article", "summary"),
    "psc": ("extract_text", "summary_text"),
    "samsum": ("dialogue", "summary"),
    "thaisum": ("body", "summary"),
    "xglue": ("news_body", "news_title"),
    "xsum": ("document", "summary"),
    "wiki_summary": ("article", "highlights"),
}


class SummarizationManager(DatasetManagerBase):
    """
    Base dataset manager for summarization tasks.
    """
    task_type = "summarization"

    def __init__(self, args: DatasetArguments, tokenizer: PreTrainedTokenizerBase = None):
        super().__init__(args, tokenizer=tokenizer, load_metric="rouge")
        self._encode_remove_columns = True
        self.text_column, self.summary_column = summarization_name_mapping[args.dataset_name]

    def encode_batch(self, examples, max_target_length=None):
        max_target_length = max_target_length or self.args.max_target_length
        padding = "max_length" if self.args.pad_to_max_length else False

        # remove pairs where at least one record is None
        inputs, targets = [], []
        for i in range(len(examples[self.text_column])):
            if examples[self.text_column][i] is not None and examples[self.summary_column][i] is not None:
                inputs.append(examples[self.text_column][i])
                targets.append(examples[self.summary_column][i])

        inputs = [self.args.source_prefix + inp for inp in inputs]
        model_inputs = self.tokenizer(inputs, max_length=self.args.max_source_length, padding=padding, truncation=True)

        # Setup the tokenizer for targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length":
            labels["input_ids"] = [
                [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def encode_batch_eval(self, examples):
        return self.encode_batch(examples, max_target_length=self.args.val_max_target_length)

    def get_data_collator(self, model=None):
        label_pad_token_id = self.tokenizer.pad_token_id
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=None,
        )

        return data_collator

    def postprocess_text(self, preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    def compute_metrics(self, predictions, references):
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # if data_args.ignore_pad_token_for_loss:
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(references != -100, references, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = self.postprocess_text(decoded_preds, decoded_labels)

        result = self.metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        # Extract a few results from ROUGE
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    def get_prediction_head_config(self):
        return {
            "head_type": "seq2seq_lm",
            "layers": 1,
        }
