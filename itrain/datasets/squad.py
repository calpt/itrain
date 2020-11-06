import logging
import os

import torch
from datasets import GenerateMode, Split, load_dataset, load_metric
from transformers import PreTrainedTokenizerBase, default_data_collator
from transformers.data.metrics.squad_metrics import compute_predictions_logits
from transformers.data.processors.squad import SquadExample, SquadResult, squad_convert_examples_to_features

from ..arguments import DatasetArguments
from .dataset_manager import CacheMode, DatasetManager


logger = logging.getLogger(__name__)


class SquadV1Manager(DatasetManager):
    label_column_names = ["start_positions", "end_positions"]
    with_negative = False

    def __init__(self, args: DatasetArguments, tokenizer: PreTrainedTokenizerBase = None):
        super().__init__(args, tokenizer)

    def load(self, cache_mode: CacheMode = CacheMode.USE_DATASET_USE_FEATURES):
        # load dataset
        download_mode = (
            GenerateMode.FORCE_REDOWNLOAD
            if cache_mode == CacheMode.NEW_DATASET_NEW_FEATURES
            else GenerateMode.REUSE_DATASET_IF_EXISTS
        )
        raw_dataset = load_dataset(
            self._dataset_loc,
            self.args.task_name,
            download_mode=download_mode,
        )
        self.examples = {}
        self.features = {}
        self.dataset = {}
        for split in [Split.TRAIN, Split.VALIDATION]:
            cache_file = self._get_features_cache_file(str(split))
            if os.path.exists(cache_file) and cache_mode >= CacheMode.USE_DATASET_USE_FEATURES:
                logger.info(f"Loading features from cache at {cache_file}.")
                feature_dict = torch.load(cache_file)
                self.examples[split] = feature_dict["examples"]
                self.features[split] = feature_dict["features"]
                self.dataset[split] = feature_dict["dataset"]
            else:
                examples = []
                for raw_example in raw_dataset[split]:
                    is_impossible = len(raw_example["answers"]["text"]) < 1
                    if not is_impossible:
                        answer_text = raw_example["answers"]["text"][0]
                        answer_start = raw_example["answers"]["answer_start"][0]
                    else:
                        answer_text, answer_start = None, None
                    examples.append(SquadExample(
                        qas_id=raw_example["id"],
                        question_text=raw_example["question"],
                        context_text=raw_example["context"],
                        answer_text=answer_text,
                        start_position_character=answer_start,
                        title=raw_example["title"],
                        answers=raw_example["answers"],
                        is_impossible=is_impossible,
                    ))
                features, dataset = squad_convert_examples_to_features(
                    examples=examples,
                    tokenizer=self.tokenizer,
                    max_seq_length=self.args.max_seq_length,
                    doc_stride=128,
                    max_query_length=64,
                    is_training=(split == Split.TRAIN),
                    return_dataset="pt",
                )
                self.examples[split] = examples
                self.features[split] = features
                self.dataset[split] = dataset
                # save to cache
                torch.save(
                    {"examples": examples, "features": features, "dataset": dataset},
                    cache_file
                )
        # load metric
        self.metric = load_metric(self.args.dataset_name, self.args.task_name)

    def collate_fn(self, features):
        batch = default_data_collator(features)
        # HACK: fixes labels for adapter-transformers qa head
        batch["labels"] = torch.stack([batch["start_positions"], batch["end_positions"]])
        del batch["start_positions"]
        del batch["end_positions"]
        return batch

    def _get_squad_results(self, predictions):
        for prediction, example in zip(predictions, self.examples[Split.VALIDATION]):
            yield SquadResult(example.qas_id, prediction[0], prediction[1])

    def compute_metrics(self, predictions, references):
        self.dev_split.reset_format()
        squad_results = self._get_squad_results(predictions)
        predictions = compute_predictions_logits(
            self.examples[Split.VALIDATION],
            self.features[Split.VALIDATION],
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
        return self.metric.compute(predictions=predictions, references=self.examples[Split.VALIDATION])

    def get_prediction_head_config(self):
        return {
            "head_type": "question_answering",
            "num_labels": 2,
            "layers": 1,
            "activation_function": "tanh",
        }


class SquadV2Manager(SquadV1Manager):
    with_negative = True
