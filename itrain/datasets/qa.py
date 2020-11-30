import logging
import os
import time
from dataclasses import asdict

import torch
from datasets import DownloadConfig, DownloadManager, load_metric
from filelock import FileLock
from transformers import PreTrainedTokenizerBase, SquadDataset, SquadDataTrainingArguments
from transformers.data.datasets.squad import Split as SquadSplit
from transformers.data.metrics.squad_metrics import compute_predictions_logits
from transformers.data.processors.squad import (
    SquadResult,
    SquadV1Processor,
    SquadV2Processor,
    squad_convert_examples_to_features,
)

from ..arguments import DatasetArguments
from .dataset_manager import DATASET_FEATURES_CACHE, CacheMode, DatasetManager


logger = logging.getLogger(__name__)


DOWNLOAD_URL_BASE = "https://multiqa.s3.amazonaws.com/squad2-0_format_data/"

DATASET_DOWNLOAD_CACHE = os.path.join(DATASET_FEATURES_CACHE, "download")
if not os.path.exists(DATASET_DOWNLOAD_CACHE):
    os.makedirs(DATASET_DOWNLOAD_CACHE)


class SquadLikeDataset(SquadDataset):
    def __init__(
        self,
        args: SquadDataTrainingArguments,
        tokenizer: PreTrainedTokenizerBase,
        cached_features_file: str,
        mode: str = "train",
        data_file: str = None,
    ):
        self.args = args
        self.processor = SquadV2Processor() if args.version_2_with_negative else SquadV1Processor()
        if isinstance(mode, str):
            try:
                mode = SquadSplit[mode]
            except KeyError:
                raise KeyError("mode is not a valid split name")
        self.mode = mode

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):
            if os.path.exists(cached_features_file) and not args.overwrite_cache:
                start = time.time()
                self.old_features = torch.load(cached_features_file)

                self.features = self.old_features["features"]
                self.examples = self.old_features.get("examples", None)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )
            else:
                if mode == SquadSplit.dev:
                    self.examples = self.processor.get_dev_examples(args.data_dir, filename=data_file)
                else:
                    self.examples = self.processor.get_train_examples(args.data_dir, filename=data_file)

                self.features = squad_convert_examples_to_features(
                    examples=self.examples,
                    tokenizer=tokenizer,
                    max_seq_length=args.max_seq_length,
                    doc_stride=args.doc_stride,
                    max_query_length=args.max_query_length,
                    is_training=mode == SquadSplit.train,
                    threads=args.threads,
                )

                start = time.time()
                torch.save(
                    {"features": self.features, "examples": self.examples},
                    cached_features_file,
                )
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )


class SquadV1Manager(DatasetManager):
    label_column_names = ["start_positions", "end_positions"]
    with_negative = False
    train_file_name = "SQuAD1-1_train.json.gz"
    dev_file_name = "SQuAD1-1_dev.json.gz"

    def __init__(self, args: DatasetArguments, tokenizer: PreTrainedTokenizerBase = None):
        super().__init__(args, tokenizer=tokenizer)
        self.train_split_name = "train"
        self.dev_split_name = "dev"
        self.always_call_metrics = True

    def _create_squad_training_arguments(self, data_dir, overwrite_cache=False):
        # copy fields from DatasetArguments
        self.squad_args = SquadDataTrainingArguments()
        for k, v in asdict(self.args).items():
            if hasattr(self.squad_args, k):
                setattr(self.squad_args, k, v)
        # set additional options manually
        self.squad_args.data_dir = data_dir
        self.squad_args.version_2_with_negative = self.with_negative
        self.squad_args.overwrite_cache = overwrite_cache

    def load(self, cache_mode: CacheMode = CacheMode.USE_DATASET_USE_FEATURES):
        # download & extract dataset
        dl_cache_dir = os.path.join(DATASET_DOWNLOAD_CACHE, self.args.identifier)
        download_config = DownloadConfig(
            cache_dir=dl_cache_dir,
            force_download=cache_mode == CacheMode.NEW_DATASET_NEW_FEATURES,
        )
        download_manager = DownloadManager(dataset_name=self.args.dataset_name, download_config=download_config)
        dl_train_file, dl_dev_file = download_manager.download_and_extract(
            [DOWNLOAD_URL_BASE + name for name in [self.train_file_name, self.dev_file_name]]
        )
        # create dataset object
        self._create_squad_training_arguments(
            os.path.dirname(dl_train_file), overwrite_cache=cache_mode < CacheMode.USE_DATASET_USE_FEATURES
        )
        self.dataset = {
            self.train_split_name: SquadLikeDataset(
                self.squad_args,
                tokenizer=self.tokenizer,
                cached_features_file=self._get_features_cache_file(self.train_split_name),
                mode="train",
                data_file=os.path.basename(dl_train_file),
            ),
            self.dev_split_name: SquadLikeDataset(
                self.squad_args,
                tokenizer=self.tokenizer,
                cached_features_file=self._get_features_cache_file(self.dev_split_name),
                mode="dev",
                data_file=os.path.basename(dl_dev_file),
            ),
        }
        # load metric
        if self.with_negative:
            self.metric = load_metric("squad_v2")
        else:
            self.metric = load_metric("squad")

    def _get_squad_results(self, predictions):
        for start_logits, end_logits, features in zip(predictions[0], predictions[1], self.dev_split.features):
            yield SquadResult(features.unique_id, start_logits, end_logits)

    def compute_metrics(self, predictions, references):
        squad_results = self._get_squad_results(predictions)
        predictions = compute_predictions_logits(
            self.dev_split.examples,
            self.dev_split.features,
            squad_results,
            n_best_size=self.squad_args.n_best_size,
            max_answer_length=self.squad_args.max_answer_length,
            do_lower_case=False,  # TODO
            output_prediction_file=None,
            output_nbest_file=None,
            output_null_log_odds_file=None,
            verbose_logging=False,
            version_2_with_negative=self.with_negative,
            null_score_diff_threshold=self.squad_args.null_score_diff_threshold,
            tokenizer=self.tokenizer,
        )
        predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]
        references = [{"id": ex.qas_id, "answers": ex.answers} for ex in self.dev_split.examples]
        scores = self.metric.compute(predictions=predictions, references=references)
        return {"em": scores["exact_match"] / 100.0, "f1": scores["f1"] / 100.0}

    def get_prediction_head_config(self):
        return {
            "head_type": "question_answering",
            "num_labels": 2,
            "layers": 1,
            "activation_function": "tanh",
        }


class SquadV2Manager(SquadV1Manager):
    with_negative = True
    train_file_name = "SQuAD2-0_train.json.gz"
    dev_file_name = "SQuAD2-0_dev.json.gz"
