import random
import time
from typing import Optional

import numpy as np
import torch
from transformers import AdapterTrainer as TransformersAdapterTrainer
from transformers import EarlyStoppingCallback, EvalPrediction, PreTrainedModel
from transformers import Seq2SeqAdapterTrainer as TransformersSeq2SeqAdapterTrainer
from transformers import Seq2SeqTrainer as TransformersSeq2SeqTrainer
from transformers import Trainer as TransformersTrainer
from transformers import is_torch_tpu_available
from transformers.trainer_utils import PredictionOutput

from .arguments import RunArguments
from .datasets import DatasetManager
from .datasets.external.utils_qa import postprocess_qa_predictions


if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met


def set_seed(seed: int):
    if seed is None:
        seed = int((time.time() * 1000) % 2**32)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


class TrainerMixin:
    model: PreTrainedModel
    dataset_manager: DatasetManager

    def __init__(
        self,
        model: PreTrainedModel,
        args: RunArguments,
        dataset_manager: DatasetManager,
        loggers=None,
    ):
        super().__init__(
            model,
            args.to_hf_training_args(self, loggers),
            data_collator=dataset_manager.get_data_collator(model),
            train_dataset=dataset_manager.train_split,
            eval_dataset=dataset_manager.dev_split,
            tokenizer=dataset_manager.tokenizer,
            compute_metrics=lambda p: dataset_manager.compute_metrics(p.predictions, p.label_ids),
        )
        self.dataset_manager = dataset_manager
        self.label_names = self.dataset_manager.label_column_names

        # early stopping
        if args.patience > 0:
            callback = EarlyStoppingCallback(args.patience)
            self.add_callback(callback)

    # Override this method because AutoModelWithHeads wouldn't work otherwise.
    def _remove_unused_columns(self, dataset, description: Optional[str] = None):
        return dataset

    def _get_train_sampler(self) -> Optional[torch.utils.data.sampler.Sampler]:
        return self.dataset_manager.train_sampler()


# Copied from https://github.com/huggingface/transformers/blob/v4.19.2/examples/pytorch/question-answering/trainer_qa.py.
class QATrainerMixin(TrainerMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # Post-processing:
    def post_process_function(self, examples, features, predictions, stage="eval"):
        # HACK: Make sure only two logits are passed to eval
        predictions = predictions[:2]
        features.reset_format()
        # Post-processing: we match the start logits and end logits to answers in the original context.
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            version_2_with_negative=self.dataset_manager.with_negative,
            n_best_size=self.dataset_manager.args.n_best_size,
            max_answer_length=self.dataset_manager.args.max_answer_length,
            null_score_diff_threshold=0.0,
            output_dir=self.args.output_dir,
            prefix=stage,
        )
        format_columns = ["input_ids"]
        if set(self.dataset_manager.label_column_names) <= set(features.column_names):
            format_columns += self.label_column_names
        features.set_format(columns=format_columns + self.dataset_manager.tokenizer.model_input_names)

        # Format the result to the format the metric expects.
        if self.dataset_manager.with_negative:
            formatted_predictions = [
                {"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
            ]
        else:
            formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]

        references = [{"id": ex["id"], "answers": ex[self.dataset_manager.answer_column_name]} for ex in examples]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    def evaluate(self, eval_dataset=None, eval_examples=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = (
            self.dataset_manager.raw_dataset[self.dataset_manager.dev_split_name]
            if eval_examples is None
            else eval_examples
        )

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        try:
            output = eval_loop(
                eval_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.compute_metrics is not None:
            eval_preds = self.post_process_function(eval_examples, eval_dataset, output.predictions)
            metrics = self.compute_metrics(eval_preds)

            # Prefix all keys with metric_key_prefix + '_'
            for key in list(metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}_"):
                    metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

            self.log(metrics)
        else:
            metrics = {}

        if self.args.tpu_metrics_debug or self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics

    def predict(self, predict_dataset, predict_examples, ignore_keys=None, metric_key_prefix: str = "test"):
        predict_dataloader = self.get_test_dataloader(predict_dataset)

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        try:
            output = eval_loop(
                predict_dataloader,
                description="Prediction",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.post_process_function is None or self.compute_metrics is None:
            return output

        predictions = self.post_process_function(predict_examples, predict_dataset, output.predictions, "predict")
        metrics = self.compute_metrics(predictions)

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return PredictionOutput(predictions=predictions.predictions, label_ids=predictions.label_ids, metrics=metrics)


class AdapterTrainer(TrainerMixin, TransformersAdapterTrainer):
    pass


class FineTuningTrainer(TrainerMixin, TransformersTrainer):
    pass


class QAAdapterTrainer(QATrainerMixin, TransformersAdapterTrainer):
    pass


class QAFineTuningTrainer(QATrainerMixin, TransformersTrainer):
    pass


class Seq2SeqAdapterTrainer(TrainerMixin, TransformersSeq2SeqAdapterTrainer):
    pass


class Seq2SeqFineTuningTrainer(TrainerMixin, TransformersSeq2SeqTrainer):
    pass


def get_trainer_class(task_type: str, is_full_finetuning: bool) -> TrainerMixin:
    if task_type == "question_answering":
        if not is_full_finetuning:
            return QAAdapterTrainer
        else:
            return QAFineTuningTrainer
    elif task_type == "summarization":
        if not is_full_finetuning:
            return Seq2SeqAdapterTrainer
        else:
            return Seq2SeqFineTuningTrainer
    else:
        if not is_full_finetuning:
            return AdapterTrainer
        else:
            return FineTuningTrainer
