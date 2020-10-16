import json
import logging
import os
import random
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm, trange
from transformers import AdamW, PreTrainedModel, get_linear_schedule_with_warmup
from transformers.adapter_bert import get_fusion_regularization_loss
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, PredictionOutput

from .arguments import RunArguments
from .datasets import DatasetManager


logger = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Runner:
    model: PreTrainedModel
    args: RunArguments
    dataset_manager: DatasetManager

    def __init__(
        self,
        model: PreTrainedModel,
        args: RunArguments,
        dataset_manager: DatasetManager,
        do_save_full_model: bool = True,
        do_save_adapters: bool = False,
        do_save_adapter_fusion: bool = False,
    ):
        self.model = model
        self.args = args
        self.dataset_manager = dataset_manager
        self.tb_writer = SummaryWriter(self.args.logging_dir)

        self.do_save_full_model = do_save_full_model
        self.do_save_adapters = do_save_adapters
        self.do_save_adapter_fusion = do_save_adapter_fusion

        set_seed(self.args.seed)

    def get_train_dataloader(self) -> DataLoader:
        if self.dataset_manager.train_split is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        data_loader = DataLoader(
            self.dataset_manager.train_split,
            batch_size=self.args.batch_size,
            shuffle=True,
            collate_fn=self.dataset_manager.collate_fn,
        )

        return data_loader

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        if eval_dataset is None and self.dataset_manager.dev_split is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        eval_dataset = eval_dataset if eval_dataset is not None else self.dataset_manager.dev_split

        data_loader = DataLoader(
            eval_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            collate_fn=self.dataset_manager.collate_fn,
        )

        return data_loader

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        data_loader = DataLoader(
            test_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            collate_fn=self.dataset_manager.collate_fn,
        )

        return data_loader

    def get_optimizers(
        self, num_training_steps: int
    ) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        if hasattr(self.model.config, "adapter_fusion_models"):
            no_decay += [f"adapter_fusion_layer.{n}.value" for n in self.model.config.adapter_fusion_models]

        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=num_training_steps
        )
        return optimizer, scheduler

    def train(self, model_path: Optional[str] = None):
        """
        Main training entry point.

        Args:
            model_path:
                (Optional) Local path to model if model to train has been instantiated from a local path
                If present, we will try reloading the optimizer/scheduler states from there.
        """
        train_dataloader = self.get_train_dataloader()
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            num_train_epochs = (
                self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
            )
        else:
            t_total = int(len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs)
            num_train_epochs = self.args.num_train_epochs

        optimizer, scheduler = self.get_optimizers(num_training_steps=t_total)

        # Check if saved optimizer or scheduler states exist
        if (
            model_path is not None
            and os.path.isfile(os.path.join(model_path, "optimizer.pt"))
            and os.path.isfile(os.path.join(model_path, "scheduler.pt"))
        ):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(
                torch.load(os.path.join(model_path, "optimizer.pt"), map_location=self.args.device)
            )
            scheduler.load_state_dict(torch.load(os.path.join(model_path, "scheduler.pt")))

        model = self.model

        if self.tb_writer is not None:
            self.tb_writer.add_text("args", self.args.to_json_string())
            self.tb_writer.add_hparams(self.args.to_sanitized_dict(), metric_dict={})

        # Train!
        total_train_batch_size = (
            self.args.batch_size
            * self.args.gradient_accumulation_steps
        )
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataloader.dataset))
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Instantaneous batch size per device = %d", self.args.batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        self.global_step = 0
        self.epoch = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        # Check if continuing training from a checkpoint
        if model_path is not None:
            # set global_step to global_step of last saved checkpoint from model path
            try:
                self.global_step = int(model_path.split("-")[-1].split("/")[0])
                epochs_trained = self.global_step // (len(train_dataloader) // self.args.gradient_accumulation_steps)
                steps_trained_in_current_epoch = self.global_step % (
                    len(train_dataloader) // self.args.gradient_accumulation_steps
                )

                logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                logger.info("  Continuing training from epoch %d", epochs_trained)
                logger.info("  Continuing training from global step %d", self.global_step)
                logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
            except ValueError:
                self.global_step = 0
                logger.info("  Starting fine-tuning.")

        tr_loss = 0.0
        logging_loss = 0.0
        best_eval_score = None
        evals_without_improvement = 0
        model.zero_grad()
        train_iterator = trange(
            epochs_trained, int(num_train_epochs), desc="Epoch"
        )
        for epoch in train_iterator:

            epoch_iterator = tqdm(train_dataloader, desc="Iteration")

            for step, inputs in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                tr_loss += self._training_step(model, inputs, optimizer)

                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    len(epoch_iterator) <= self.args.gradient_accumulation_steps
                    and (step + 1) == len(epoch_iterator)
                ):
                    # apply adapter fusion weight regularization on the value matrix
                    if (
                        hasattr(self.model.config, "adapter_fusion")
                        and self.model.config.adapter_fusion["regularization"]
                    ):
                        fusion_reg_loss = get_fusion_regularization_loss(self.model)
                        fusion_reg_loss.backward()

                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

                    optimizer.step()

                    scheduler.step()
                    model.zero_grad()
                    self.global_step += 1
                    self.epoch = epoch + (step + 1) / len(epoch_iterator)

                if ((self.args.max_steps > 0 and self.global_step > self.args.max_steps)):
                    epoch_iterator.close()
                    break

            # Log epoch
            logs: Dict[str, float] = {}
            logs["loss"] = (tr_loss - logging_loss) / len(train_dataloader)
            logs["learning_rate"] = scheduler.get_last_lr()[0]
            logging_loss = tr_loss

            self._log(logs)

            # Save checkpoint
            if self.args.checkpoint_epochs > 0 and self.epoch % self.args.checkpoint_epochs == 0:
                # In all cases (even distributed/parallel), self.model is always a reference
                # to the model we want to save.
                if hasattr(model, "module"):
                    assert model.module is self.model
                else:
                    assert model is self.model
                # Save model checkpoint
                output_dir = os.path.join(self.args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{self.global_step}")

                self.save_model(output_dir)
                self._rotate_checkpoints()

                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

            # Evaluate epoch
            if self.args.evaluate_during_training:
                results = self.evaluate()
                if self.args.patience > 0:
                    # Keep track of best loss to determine if we should stop early
                    eval_score = results[self.args.patience_metric]
                    if not best_eval_score or self._compare_scores(eval_score, best_eval_score):
                        evals_without_improvement = 0
                        best_eval_score = eval_score

                        # Save checkpoint of best model
                        if hasattr(model, "module"):
                            assert model.module is self.model
                        else:
                            assert model is self.model
                        # Save model checkpoint
                        output_dir = os.path.join(self.args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-best")

                        self.save_model(output_dir)

                        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

                        # Save eval results
                        with open(os.path.join(output_dir, "results.json"), "w") as f:
                            json.dump(results, f)
                    else:
                        evals_without_improvement += 1
                        if evals_without_improvement >= self.args.patience:
                            logger.info(
                                f"Early stopping threshold ({self.args.patience}) exceeded, stopping training"
                            )
                # log eval results
                logger.info("***** Eval results {} *****".format(self.dataset_manager.name))
                for key, value in results.items():
                    logger.info("  %s = %s", key, value)

            if ((self.args.max_steps > 0 and self.global_step > self.args.max_steps) or (
                    self.args.patience > 0 and evals_without_improvement >= self.args.patience)):
                train_iterator.close()
                break

        if self.tb_writer:
            self.tb_writer.close()

        logger.info("Training completed.")
        return self.global_step, tr_loss / self.global_step, best_eval_score

    def _log(self, logs: Dict[str, float], iterator: Optional[tqdm] = None) -> None:
        if self.epoch is not None:
            logs["epoch"] = self.epoch
        if self.tb_writer:
            for k, v in logs.items():
                self.tb_writer.add_scalar(k, v, self.epoch)
        output = json.dumps({**logs, **{"step": self.global_step}})
        if iterator is not None:
            iterator.write(output)
        else:
            print(output)

    def _compare_scores(self, current, best):
        if "loss" in self.args.patience_metric:
            return current < best
        else:
            return current > best

    def _training_step(
        self, model: nn.Module, inputs: Dict[str, torch.Tensor], optimizer: torch.optim.Optimizer
    ) -> float:
        model.train()
        for k, v in inputs.items():
            inputs[k] = v.to(self.args.device)

        outputs = model(**inputs)
        loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        loss.backward()

        return loss.item()

    def save_model(self):
        output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        if self.do_save_adapters:
            self.model.save_all_adapters(output_dir)
        if self.do_save_adapter_fusion:
            self.model.save_all_adapter_fusions(output_dir)
        if self.do_save_full_model:
            self.model.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def _sorted_checkpoints(self, checkpoint_prefix=PREFIX_CHECKPOINT_DIR, use_mtime=False) -> List[str]:
        ordering_and_checkpoint_path = []

        glob_checkpoints = [str(x) for x in Path(self.args.output_dir).glob(f"{checkpoint_prefix}-*")]

        for path in glob_checkpoints:
            if use_mtime:
                ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
            else:
                regex_match = re.match(f".*{checkpoint_prefix}-([0-9]+)", path)
                if regex_match and regex_match.groups():
                    ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

        checkpoints_sorted = sorted(ordering_and_checkpoint_path)
        checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
        return checkpoints_sorted

    def _rotate_checkpoints(self, use_mtime=False) -> None:
        if self.args.save_total_limit is None or self.args.save_total_limit <= 0:
            return

        # Check if we should delete older checkpoint(s)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=use_mtime)
        if len(checkpoints_sorted) <= self.args.save_total_limit:
            return

        number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - self.args.save_total_limit)
        checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
        for checkpoint in checkpoints_to_be_deleted:
            logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
            shutil.rmtree(checkpoint)

    def evaluate(
        self, eval_dataset: Optional[Dataset] = None, prediction_loss_only: Optional[bool] = None,
    ) -> Dict[str, float]:
        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        output = self._prediction_loop(eval_dataloader, description="Evaluation")

        self._log(output.metrics)

        return output.metrics

    def predict(self, test_dataset: Dataset) -> PredictionOutput:
        test_dataloader = self.get_test_dataloader(test_dataset)

        return self._prediction_loop(test_dataloader, description="Prediction")

    def _prediction_loop(
        self, dataloader: DataLoader, description: str,
    ) -> PredictionOutput:
        """
        Prediction/evaluation loop, shared by `evaluate()` and `predict()`.

        Works both with or without labels.
        """
        model = self.model

        batch_size = dataloader.batch_size
        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", len(dataloader.dataset))
        logger.info("  Batch size = %d", batch_size)
        eval_losses: List[float] = []
        preds: torch.Tensor = None
        label_ids: torch.Tensor = None
        model.eval()

        for inputs in tqdm(dataloader, desc=description):
            has_labels = any(inputs.get(k) is not None for k in ["labels", "lm_labels", "masked_lm_labels"])

            for k, v in inputs.items():
                inputs[k] = v.to(self.args.device)

            with torch.no_grad():
                outputs = model(**inputs)
                if has_labels:
                    step_eval_loss, logits = outputs[:2]
                    eval_losses += [step_eval_loss.mean().item()]
                else:
                    logits = outputs[0]

            if preds is None:
                preds = logits.detach()
            else:
                preds = torch.cat((preds, logits.detach()), dim=0)
            if inputs.get("labels") is not None:
                if label_ids is None:
                    label_ids = inputs["labels"].detach()
                else:
                    label_ids = torch.cat((label_ids, inputs["labels"].detach()), dim=0)

        # Finally, turn the aggregated tensors into numpy arrays.
        if preds is not None:
            preds = preds.cpu().numpy()
        if label_ids is not None:
            label_ids = label_ids.cpu().numpy()

        if self.dataset_manager.metric is not None and preds is not None and label_ids is not None:
            metrics = self.dataset_manager.compute_metrics(preds, label_ids)
        else:
            metrics = {}
        if len(eval_losses) > 0:
            metrics["eval_loss"] = np.mean(eval_losses)

        # Prefix all keys with eval_
        for key in list(metrics.keys()):
            if not key.startswith("eval_"):
                metrics[f"eval_{key}"] = metrics.pop(key)

        return PredictionOutput(predictions=preds, label_ids=label_ids, metrics=metrics)