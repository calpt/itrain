from typing import List

import torch
from transformers import PreTrainedTokenizerBase, default_data_collator
from transformers.data.metrics.squad_metrics import compute_predictions_logits
from transformers.data.processors.squad import SquadExample, SquadFeatures, SquadResult

from ..arguments import DatasetArguments
from .dataset_manager import ColumnConfig, DatasetManager


class SquadV1Manager(DatasetManager):
    label_column_names = ["start_positions", "end_positions"]
    with_negative = False

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
        encoded.update({"start_positions": start_positions, "end_positions": end_positions})
        return encoded

    def collate_fn(self, features):
        batch = default_data_collator(features)
        # HACK: fixes labels for adapter-transformers qa head
        batch["labels"] = torch.stack([batch["start_positions"], batch["end_positions"]])
        del batch["start_positions"]
        del batch["end_positions"]
        return batch

    def _get_squad_examples_and_features(self, dataset):
        squad_examples, squad_features = [], []
        for example in dataset:
            squad_examples.append(SquadExample(
                example["id"],
                example["question"],
                example["context"],
                answer_text=None,
                start_position_character=None,
                title=example["title"],
                answers=example["answers"],
            ))
            squad_features.append(SquadFeatures(
                input_ids=example["input_ids"],
                attention_mask=example["attention_mask"],
                token_type_ids=example.get("token_type_ids", None),
                cls_index=None,
                p_mask=None,
                example_index=0,
                unique_id=example["id"],
                paragraph_len=None,
                token_is_max_context=None,
                tokens=None,
                token_to_orig_map=None,
                start_position=None,
                end_position=None,
                is_impossible=None,
                qas_id=example["id"],
            ))
        return squad_examples, squad_features

    def _get_squad_results(self, predictions, examples: List[SquadExample]):
        for prediction, example in zip(predictions, examples):
            yield SquadResult(example.qas_id, prediction[0], prediction[1])

    def compute_metrics(self, predictions, references):
        self.dev_split.reset_format()
        squad_examples, squad_features = self._get_squad_examples_and_features(self.dev_split)
        squad_results = self._get_squad_results(predictions, squad_examples)
        predictions = compute_predictions_logits(
            squad_examples,
            squad_features,
            squad_results,
            n_best_size=20,
            max_answer_length=30,
            do_lower_case=True,  # TODO
            output_prediction_file=None,
            output_nbest_file=None,
            output_null_log_odds_file=None,
            verbose_logging=False,
            version_2_with_negative=self.with_negative,
            null_score_diff_threshold=0.0,
            tokenizer=self.tokenizer,
        )
        return self.metric.compute(predictions=predictions, references=squad_examples)

    def get_prediction_head_config(self):
        return {
            "head_type": "question_answering",
            "num_labels": 2,
            "layers": 1,
            "activation_function": "tanh",
        }


class SquadV2Manager(SquadV1Manager):
    with_negative = True
