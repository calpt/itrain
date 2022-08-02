import dataclasses
import json
import logging
import os
import time
from collections import defaultdict
from typing import List, Optional, Sequence, Union

import numpy as np
import transformers
import wandb
from ruamel.yaml import YAML
from transformers import PreTrainedModel

from .arguments import DatasetArguments, ModelArguments, RunArguments
from .datasets import DATASET_MANAGER_CLASSES, DatasetManager
from .model_creator import create_model, create_tokenizer, register_heads
from .notifier import NOTIFIER_CLASSES
from .trainer import get_trainer_class, set_seed


OUTPUT_FOLDER = os.environ.get("ITRAIN_OUTPUT") or "run_output"
LOGS_FOLDER = os.environ.get("ITRAIN_LOGS") or "run_logs"

SETUP_OUTPUT_FILE = "itrain_setup.json"
STATE_OUTPUT_FILE = "itrain_state.json"

logging.basicConfig(level=logging.INFO)
transformers.logging.set_verbosity_info()
logger = logging.getLogger(__name__)


class Setup:
    """
    Defines, configures and runs a full Transformer training setup,
    including dataset configuration, model setup, training and evaluation.
    """

    id: int
    name: str
    dataset_manager: DatasetManager
    model_instance: PreTrainedModel
    model_args: ModelArguments

    def __init__(self, id=0, name=None):
        self.id = id
        self.name = name
        self.dataset_manager = None
        self.model_instance = None
        self.model_args = None
        self._train_run_args = None
        self._do_eval = False
        self._eval_run_args = None
        self._eval_split = None
        self._loggers = {}
        self._notifiers = {}
        self.restarts = None

    @property
    def full_name(self) -> List[str]:
        name = self.name or self.dataset_manager.name
        return [self.model_instance.config.model_type, name, str(self.id)]

    @property
    def output_dir(self) -> str:
        return self._eval_run_args.output_dir if self._eval_run_args else self._train_run_args.output_dir

    def dataset(self, args_or_manager: Union[DatasetArguments, DatasetManager]):
        """
        Set up the dataset.
        """
        if self.dataset_manager is not None:
            raise ValueError("Dataset already set.")
        if isinstance(args_or_manager, DatasetManager):
            self.dataset_manager = args_or_manager
        else:
            self.dataset_manager = DATASET_MANAGER_CLASSES[args_or_manager.dataset_name](args_or_manager)

    def model(self, args: ModelArguments):
        """
        Set up the model.
        """
        if self.model_instance is not None:
            raise ValueError("Model already set.")
        self.model_args = args

    def training(self, args: RunArguments):
        """
        Set up training.
        """
        self._train_run_args = args

    def evaluation(self, args: Optional[RunArguments] = None, split=None):
        """
        Set up evaluation.
        """
        self._do_eval = True
        self._eval_run_args = args
        self._eval_split = split

    def logging(self, logger_name: str, **kwargs):
        """
        Set up logging to Tensorboard or W&B.
        """
        self._loggers[logger_name] = kwargs

    def notify(self, notifier_name: str, **kwargs):
        """
        Set up a notifier. Can be either "telegram" or "email" currently.
        """
        self.notifiers.append(NOTIFIER_CLASSES[notifier_name](**kwargs))

    def _override(self, instance, overrides):
        orig_dict = dataclasses.asdict(instance)
        for k in orig_dict.keys():
            if k in overrides:
                v = overrides[k]
                if v is not None:
                    orig_dict[k] = v
        return type(instance)(**orig_dict)

    @classmethod
    def from_dict(cls, config, overrides=None):
        setup = cls(id=config.get("id", 0), name=config.get("name", None))
        dataset_args = DatasetArguments(**config["dataset"])
        if overrides:
            dataset_args = setup._override(dataset_args, overrides)
        setup.dataset(dataset_args)
        model_args = ModelArguments(**config["model"])
        if overrides:
            model_args = setup._override(model_args, overrides)
        setup.model(model_args)
        if "training" in config:
            train_args = RunArguments(**config["training"])
            if overrides:
                train_args = setup._override(train_args, overrides)
            setup.training(train_args)
        if "evaluation" in config:
            if isinstance(config["evaluation"], dict):
                eval_args = RunArguments(**config["evaluation"])
                if overrides:
                    eval_args = setup._override(eval_args, overrides)
                setup.evaluation(eval_args)
            elif isinstance(config["evaluation"], str):
                setup.evaluation(split=config["evaluation"])
            elif config["evaluation"]:
                setup.evaluation()
        if "logging" in config:
            for logger_name, kwargs in config["logging"].items():
                setup.logging(logger_name, **kwargs)
        if "notify" in config:
            for notifier_name, kwargs in config["notify"].items():
                setup.notify(notifier_name, **kwargs)
        setup.restarts = config["restarts"]

        return setup

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "dataset": self.dataset_manager.args.to_dict(),
            "model": self.model_args.to_dict(),
            "training": self._train_run_args.to_dict() if self._train_run_args else None,
            "evaluation": self._eval_run_args.to_dict() if self._eval_run_args else (self._eval_split or self._do_eval),
            "logging": self._loggers,
            "notify": {k: v.to_dict() for k, v in self._notifiers.items()},
            "restarts": self.restarts,
        }

    @classmethod
    def from_file(cls, file, overrides=None):
        """
        Load a setup from file.
        """
        with open(file, "r", encoding="utf-8") as f:
            if file.endswith(".yaml") or file.endswith(".yml"):
                yaml = YAML(typ="safe")
                config = yaml.load(f)
            else:
                config = json.load(f)
        if "name" not in config:
            config["name"] = os.path.splitext(os.path.basename(file))[0]

        return cls.from_dict(config, overrides)

    def to_file(self, file):
        """
        Save the setup to file.
        """
        with open(file, "w", encoding="utf-8") as f:
            if file.endswith(".yaml") or file.endswith(".yml"):
                yaml = YAML()
                yaml.dump(self.to_dict(), f)
            else:
                json.dump(self.to_dict(), f, indent=4)

    def _setup_tokenizer(self):
        self.dataset_manager.tokenizer = create_tokenizer(
            self.model_args,
            **self.dataset_manager.get_tokenizer_config_kwargs(),
        )

    def _init_notifiers(self, num_runs=1):
        # notifier_name should not start with a numeric char
        if isinstance(self.id, int) or self.id[0].isnumeric():
            notifier_name = ["#id" + str(self.id)]
        else:
            notifier_name = [self.id]
        notifier_name.append(self.model_args.model_name_or_path)
        if self.model_args.train_adapter:
            notifier_name.append(self.model_args.adapter_config)
        notifier_name.append(self.name or self.dataset_manager.name)
        for notifier in self._notifiers.values():
            if not notifier.title:
                notifier.title = ", ".join(notifier_name)

        if num_runs > 1:
            for notifier in self._notifiers.values():
                notifier.notify_start(
                    message=f"Training setup ({num_runs} runs):", **self._train_run_args.to_sanitized_dict()
                )

    def _init_loggers(self, trial=None, do_resume=False):
        for name, kwargs in self._loggers.items():
            if name == "tensorboard":
                if "logging_dir" not in kwargs:
                    kwargs["logging_dir"] = os.path.join(LOGS_FOLDER, "_".join(self.full_name))
                    if trial is not None:
                        kwargs["logging_dir"] = os.path.join(kwargs["logging_dir"], str(trial))
            elif name == "wandb":
                wandb_kwargs = kwargs.copy()
                if "name" not in wandb_kwargs:
                    wandb_kwargs["name"] = "-".join(self.full_name)
                    if trial is not None:
                        wandb_kwargs["group"] = wandb_kwargs["name"]
                        wandb_kwargs["name"] = wandb_kwargs["name"] + f"-{trial}"
                wandb_kwargs["resume"] = do_resume
                wandb_kwargs["id"] = wandb_kwargs["name"]
                while True:
                    try:
                        wandb.init(**wandb_kwargs)
                        break
                    except Exception:
                        print("Retrying wandb.init...")
                        time.sleep(20)
            else:
                raise ValueError(f"Unknown logger name: {name}.")

    def _init_args(self):
        if self._train_run_args is not None:
            if not self._train_run_args.output_dir:
                self._train_run_args.output_dir = os.path.join(OUTPUT_FOLDER, *self.full_name)
        if self._eval_run_args is not None:
            if not self._eval_run_args.output_dir:
                self._eval_run_args.output_dir = os.path.join(OUTPUT_FOLDER, *self.full_name)

    def _prepare_run_args(self, args: RunArguments, trial=None, overrides=None):
        # create a copy with the run-specific adjustments
        prepared_args = RunArguments(**dataclasses.asdict(args))
        if trial is not None:
            prepared_args.output_dir = os.path.join(args.output_dir, str(trial))
        if overrides is not None:
            prepared_args = self._override(prepared_args, overrides)
        return prepared_args

    def run(self, restarts=None, first_run_index=0):
        """
        Run this setup. Dataset, model, and training or evaluation are expected to be set.

        Args:
            restarts (Union[list, int], optional): Defines the random training restarts. Can be either:
                - a list of integers: run training with each of the given values as random seed.
                - a single integer: number of random restarts, each with a random seed.
                - None (default): one random restart.
            first_run_index (int, optional): The start index in the sequence of random restarts. Defaults to 0.

        Returns:
            dict: A dictionary of run results containing the used random seeds and (optionally) evaluation results.
        """
        # Set up tokenizer
        self._setup_tokenizer()
        # Load dataset
        self.dataset_manager.load_and_preprocess()

        restarts = restarts or self.restarts
        if not restarts:
            restarts = [42]
        if not isinstance(restarts, Sequence):
            restarts = [None for _ in range(int(restarts))]
        # Init state
        state = {
            "restarts": restarts,
            "all_results": defaultdict(list),
            "current_trial": first_run_index,
            "trial_running": False,
        }
        # Init notifiers
        self._init_notifiers(num_runs=len(restarts))

        for i, seed in enumerate(restarts, start=first_run_index):
            self._run_trial(state, i, seed)

        # post-process aggregated results
        if len(restarts) > 1 and self._do_eval:
            self._generate_stats(state)

        return dict(state["all_results"])

    def resume(self, file, add_restarts=None):
        """
        Resume running this setup from a given run state.

        Args:
            file (str): The run state file to load.

        Returns:
            dict: A dictionary of run results containing the used random seeds and (optionally) evaluation results.
        """
        # Set up tokenizer
        self._setup_tokenizer()
        # Load dataset
        self.dataset_manager.load_and_preprocess()

        if isinstance(add_restarts, int):
            restarts = [None for _ in range(add_restarts)]

        # Load run state
        with open(file, "r") as f:
            state = json.load(f)
            restarts = state["restarts"]

            if add_restarts is not None:
                restarts = restarts + add_restarts

        # Init notifiers
        self._init_notifiers(num_runs=len(restarts))

        start_index = state["current_trial"]
        for i, seed in enumerate(restarts[start_index:], start=start_index):
            do_resume = i == state["current_trial"] and state["trial_running"]
            self._run_trial(state, i, seed, do_resume=do_resume)

        # post-process aggregated results
        if len(restarts) > 1 and self._do_eval:
            self._generate_stats(state)

        return dict(state["all_results"])

    def _run_trial(self, state: dict, i: int, seed=None, do_resume=False):
        # Init & set seed
        has_restarts = len(state["restarts"]) > 1
        best_model_dir = None
        seed = set_seed(seed)

        # Set up model
        is_full_finetuning = not self.model_args.train_adapter and self.model_args.train_adapter_fusion is None
        self.model_instance = create_model(
            self.model_args,
            self.dataset_manager,
            use_classic_model_class=is_full_finetuning,
        )

        # Modify & save state
        self._init_args()  # To make sure output dirs are set
        state["all_results"]["seeds"].append(seed)
        state["current_trial"] = i
        state["trial_running"] = True
        os.makedirs(self.output_dir, exist_ok=True)
        self.to_file(os.path.join(self.output_dir, SETUP_OUTPUT_FILE))
        with open(os.path.join(self.output_dir, STATE_OUTPUT_FILE), "w") as f:
            json.dump(state, f)

        # Init loggers
        self._init_loggers(trial=i, do_resume=do_resume)

        # Configure and run training
        if self._train_run_args:
            train_run_args = self._prepare_run_args(self._train_run_args, trial=i if has_restarts else None)
            trainer_class = get_trainer_class(self.dataset_manager.task_type, is_full_finetuning)
            trainer = trainer_class(
                self.model_instance,
                train_run_args,
                self.dataset_manager,
            )
            if not has_restarts:
                for notifier in self._notifiers.values():
                    notifier.notify_start(
                        message="Training setup:", seed=seed, **train_run_args.to_sanitized_dict()
                    )
            try:
                do_resume &= len(os.listdir(train_run_args.output_dir)) > 0
                step, loss, _ = trainer.train(
                    self.model_args.model_name_or_path
                    if os.path.isdir(self.model_args.model_name_or_path)
                    else do_resume,
                )
                epoch = trainer.state.epoch
                best_score = trainer.state.best_metric
                best_model_dir = trainer.state.best_model_checkpoint
                # only save the final model if we don't use early stopping
                if train_run_args.patience <= 0:
                    trainer.save_model()
            except Exception as ex:
                for notifier in self._notifiers.values():
                    notifier.notify_error(f"{ex.__class__.__name__}: {ex}")
                raise ex
            # if no evaluation is done, we're at the end here
            if not self._do_eval:
                for notifier in self._notifiers.values():
                    notifier.notify_end(
                        message="Training results:",
                        step=step,
                        training_epochs=epoch,
                        loss=loss,
                        best_score=best_score,
                    )
            # otherwise, reload the best model for evaluation
            elif best_model_dir:
                logger.info("Reloading best model for evaluation.")
                if self.model_args.train_adapter:
                    for adapter_name in self.model_instance.config.adapters.adapters:
                        path = os.path.join(best_model_dir, adapter_name)
                        self.model_instance.load_adapter(path)
                if self.model_args.train_adapter_fusion is not None:
                    path = os.path.join(best_model_dir, self.model_args.train_adapter_fusion)
                    self.model_instance.load_adapter_fusion(path)
                    # HACK: also reload the prediction head
                    head_path = os.path.join(best_model_dir, self.dataset_manager.name)
                    self.model_instance.load_head(head_path)
                if is_full_finetuning:
                    self.model_instance = self.model_instance.from_pretrained(best_model_dir)
                    register_heads(self.model_instance)
                    self.model_instance.active_head = self.dataset_manager.name
        else:
            epoch = None

        # Configure and run eval
        if self._do_eval:
            eval_run_args = self._prepare_run_args(
                self._eval_run_args or self._train_run_args, trial=i if has_restarts else None
            )
            trainer_class = get_trainer_class(self.dataset_manager.task_type, is_full_finetuning)
            trainer = trainer_class(
                self.model_instance,
                eval_run_args,
                self.dataset_manager,
            )
            try:
                eval_dataset = self.dataset_manager.dataset[self._eval_split] if self._eval_split else None
                results = trainer.evaluate(eval_dataset=eval_dataset)
            except Exception as ex:
                for notifier in self._notifiers.values():
                    notifier.notify_error(f"{ex.__class__.__name__}: {ex}")
                raise ex
            if epoch:
                results["training_epochs"] = epoch
            if self._eval_split:
                results["eval_split"] = self._eval_split
            if best_model_dir:
                results["best_model_dir"] = best_model_dir
            output_eval_file = os.path.join(eval_run_args.output_dir, "eval_results.txt")
            with open(output_eval_file, "w") as f:
                logger.info(
                    "***** Eval results {} (trial {}) *****".format(self.name or self.dataset_manager.name, i)
                )
                for key, value in results.items():
                    logger.info("  %s = %s", key, value)
                    f.write("%s = %s\n" % (key, value))
                    # also append to aggregated results
                    state["all_results"][key].append(value)
            if not has_restarts:
                for notifier in self._notifiers.values():
                    notifier.notify_end(message="Evaluation results:", **results)

        # Save state
        state["current_trial"] = state["current_trial"] + 1
        state["trial_running"] = False
        with open(os.path.join(self.output_dir, STATE_OUTPUT_FILE), "w") as f:
            json.dump(state, f)

        # Delete model
        del self.model_instance
        # Finish wandb
        if "wandb" in self._loggers:
            wandb.finish()

    def _generate_stats(self, state: dict):
        stats = {}
        for key, values in state["all_results"].items():
            if isinstance(values[0], float):
                stats[f"{key}_avg"] = np.mean(values)
                stats[f"{key}_std"] = np.std(values)

        with open(os.path.join(self.output_dir, f"aggregated_results_{self.id}.json"), "w") as f:
            json.dump(stats, f)
        for notifier in self._notifiers.values():
            notifier.notify_end(message=f"Aggregated results ({len(state['restarts'])} runs):", **stats)
