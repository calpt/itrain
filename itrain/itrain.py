import json
import logging
import os
from typing import Optional, Union

from transformers import PreTrainedModel

from .arguments import DatasetArguments, ModelArguments, RunArguments
from .datasets import DATASET_MANAGER_CLASSES, CacheMode, DatasetManager
from .datasets.tagging import TaggingDatasetManager
from .model_creator import create_model, create_tokenizer
from .notifier import NOTIFIER_CLASSES
from .runner import Runner, set_seed


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Setup:
    id: int
    dataset_manager: DatasetManager
    model_instance: PreTrainedModel
    model_args: ModelArguments

    def __init__(self, id=0):
        self.id = id
        self.dataset_manager = None
        self.model_instance = None
        self.model_args = None
        self._train_run_args = None
        self._do_eval = False
        self._eval_run_args = None
        self._notifiers = {}
        self._config_name = None

    @property
    def name(self):
        return self._config_name or self.dataset_manager.name

    def dataset(self, args_or_manager: Union[DatasetArguments, DatasetManager]):
        if self.dataset_manager is not None:
            raise ValueError("Dataset already set.")
        if isinstance(args_or_manager, DatasetManager):
            self.dataset_manager = args_or_manager
        else:
            self.dataset_manager = DATASET_MANAGER_CLASSES[args_or_manager.dataset_name](args_or_manager)

    def model(self, args: ModelArguments):
        if self.model_instance is not None:
            raise ValueError("Model already set.")
        if not self.dataset_manager:
            raise ValueError("Set dataset before creating model.")
        self.model_args = args
        # HACK: when using e.g. Roberta with sequence tagging, we need to set add_prefix_space
        kwargs = {}
        if isinstance(self.dataset_manager, TaggingDatasetManager):
            kwargs["add_prefix_space"] = True
        self.dataset_manager.tokenizer = create_tokenizer(args, **kwargs)
        self.model_instance = create_model(args, self.dataset_manager)

    def training(self, args: RunArguments):
        self._train_run_args = args

    def evaluation(self, args: Optional[RunArguments] = None):
        self._do_eval = True
        self._eval_run_args = args

    def notify(self, notifier_name: str, **kwargs):
        self._notifiers[notifier_name] = NOTIFIER_CLASSES[notifier_name](**kwargs)

    def load_from_file(self, file):
        with open(file, "r", encoding="utf-8") as f:
            config = json.load(f)
        self.dataset(DatasetArguments(**config["dataset"]))
        self.model(ModelArguments(**config["model"]))
        if "training" in config:
            self.training(RunArguments(**config["training"]))
        if "evaluation" in config:
            if isinstance(config["evaluation"], dict):
                self.evaluation(RunArguments(**config["evaluation"]))
            elif config["evaluation"]:
                self.evaluation()
        if "notify" in config:
            self.notify(config["notify"])
        self._config_name = os.path.splitext(os.path.basename(file))[0]

    def _auto_fill_dirs(self, args: RunArguments):
        if not args.output_dir:
            args.output_dir = os.path.join(
                "run_output", self.model_instance.config.model_type, self.name, str(self.id)
            )
        if not args.logging_dir:
            args.logging_dir = os.path.join(
                "run_logs", "_".join([str(self.id), self.model_instance.config.model_type, self.name])
            )

    def run(self):
        if self._train_run_args:
            set_seed(self._train_run_args.seed)
        else:
            set_seed(self._eval_run_args.seed)
        # Load dataset
        self.dataset_manager.load_and_preprocess()
        # Init notifier
        name = ["#id" + str(self.id)]
        if self.model_instance.model_name:
            name.append(self.model_instance.model_name)
        if self.name:
            name.append(self.name)
        for notifier in self._notifiers.values():
            if not notifier.title:
                notifier.title = ", ".join(name)

        # Configure and run training
        if self._train_run_args:
            self._auto_fill_dirs(self._train_run_args)
            runner = Runner(
                self.model_instance,
                self._train_run_args,
                self.dataset_manager,
                do_save_full_model=not self.model_args.train_adapter and self.model_args.train_adapter_fusion is None,
                do_save_adapters=self.model_args.train_adapter,
                do_save_adapter_fusion=self.model_args.train_adapter_fusion is not None,
            )
            for notifier in self._notifiers.values():
                notifier.notify_start(message="Training setup:", **self._train_run_args.to_sanitized_dict())
            try:
                step, epoch, loss, best_score, best_model_dir = runner.train(
                    self.model_args.model_name_or_path if os.path.isdir(self.model_args.model_name_or_path) else None
                )
            except Exception as ex:
                for notifier in self._notifiers.values():
                    notifier.notify_error(f"{ex.__class__.__name__}: {ex}")
                raise ex
            # if no evaluation is done, we're at the end here
            if not self._do_eval:
                for notifier in self._notifiers.values():
                    notifier.notify_end(message="Training results:", step=step, training_epochs=epoch, loss=loss, best_score=best_score)
            # otherwise, reload the best model for evaluation
            elif best_model_dir:
                logger.info("Reloading best model for evaluation.")
                if self.model_args.train_adapter:
                    for adapter_name in self.model_instance.config.adapters.adapters:
                        path = os.path.join(best_model_dir, adapter_name)
                        self.model_instance.load_adapter(path)
                elif self.model_args.train_adapter_fusion is not None:
                    path = os.path.join(best_model_dir, self.model_args.train_adapter_fusion)
                    # HACK: adapter-transformers refuses to overwrite existing adapter_fusion config
                    del self.model_instance.config.adapter_fusion
                    self.model_instance.load_adapter_fusion(path)
                    # HACK: also reload the prediction head
                    head_path = os.path.join(best_model_dir, self.dataset_manager.name)
                    self.model_instance.load_head(head_path)
                else:
                    self.model_instance = self.model_instance.from_pretrained(best_model_dir)
                    self.model_instance.active_head = self.dataset_manager.name
        else:
            epoch = None

        # Configure and run eval
        if self._do_eval:
            eval_run_args = self._eval_run_args or self._train_run_args
            self._auto_fill_dirs(eval_run_args)
            runner = Runner(
                self.model_instance,
                eval_run_args,
                self.dataset_manager,
                do_save_full_model=not self.model_args.train_adapter and self.model_args.train_adapter_fusion is None,
                do_save_adapters=self.model_args.train_adapter,
                do_save_adapter_fusion=self.model_args.train_adapter_fusion is not None,
            )
            try:
                results = runner.evaluate(log=False)
                output_eval_file = os.path.join(
                    eval_run_args.output_dir, f"eval_results_{self.name}.txt"
                )
            except Exception as ex:
                for notifier in self._notifiers.values():
                    notifier.notify_error(f"{ex.__class__.__name__}: {ex}")
                raise ex
            with open(output_eval_file, "w") as f:
                logger.info("***** Eval results {} *****".format(self.name))
                for key, value in results.items():
                    logger.info("  %s = %s", key, value)
                    f.write("%s = %s\n" % (key, value))
            if epoch:
                results["training_epochs"] = epoch
            for notifier in self._notifiers.values():
                notifier.notify_end(message="Evaluation results:", **results)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Simple tool to setup Transformers training runs.")
    parser.add_argument("config", type=str, help="Path to the json file containing the full training setup.")
    parser.add_argument("--id", type=int, default=0, help="ID of this run.")
    parser.add_argument("--preprocess_only", action="store_true", default=False, help="Only run dataset preprocessing.")
    args = parser.parse_args()

    # Load and run
    setup = Setup(id=args.id)
    setup.load_from_file(args.config)
    if args.preprocess_only:
        setup.dataset_manager.load_and_preprocess(CacheMode.USE_DATASET_NEW_FEATURES)
    else:
        setup.run()


if __name__ == "__main__":
    main()
