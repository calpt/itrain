import json
import logging
import os
from typing import Union

from transformers import PreTrainedModel

from .arguments import DatasetArguments, ModelArguments, RunArguments
from .dataset_manager import DATASET_MANAGER_CLASSES, DatasetManager
from .model_creator import create_model, create_tokenizer
from .notifier import NOTIFIER_CLASSES, Notifier
from .runner import Runner, set_seed


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Setup:
    id: int
    dataset_manager: DatasetManager
    model_instance: PreTrainedModel
    model_args: ModelArguments
    notifier: Notifier

    def __init__(self, id=0):
        self.id = id
        self.dataset_manager = None
        self.model_instance = None
        self.model_args = None
        self._train_run_args = None
        self._eval_run_args = None
        self.notifier = None

    def dataset(self, args_or_manager: Union[DatasetArguments, DatasetManager]):
        if isinstance(args_or_manager, DatasetManager):
            self.dataset_manager = args_or_manager
        else:
            self.dataset_manager = DATASET_MANAGER_CLASSES[args_or_manager.dataset_name](args_or_manager)

    def model(self, args: ModelArguments):
        if not self.dataset_manager:
            raise ValueError("Set dataset before creating model.")
        self.model_args = args
        self.dataset_manager.tokenizer = create_tokenizer(args)
        self.model_instance = create_model(args, self.dataset_manager)

    def training(self, args: RunArguments):
        self._train_run_args = args

    def evaluation(self, args: RunArguments):
        self._eval_run_args = args

    def notify(self, notifier_name: str, **kwargs):
        self.notifier = NOTIFIER_CLASSES[notifier_name](**kwargs)

    def load_from_file(self, file):
        with open(file, "r", encoding="utf-8") as f:
            config = json.load(f)
        self.dataset(DatasetArguments(**config["dataset"]))
        self.model(ModelArguments(**config["model"]))
        if "training" in config:
            self.training(RunArguments(**config["training"]))
        if "evaluation" in config:
            self.evaluation(RunArguments(**config["evaluation"]))
        if "notify" in config:
            self.notify(config["notify"])

    def _auto_fill_dirs(self, args: RunArguments):
        if not args.output_dir:
            args.output_dir = os.path.join(
                "run_output", self.model_instance.config.model_type, self.dataset_manager.name, str(self.id)
            )
        if not args.logging_dir:
            args.logging_dir = os.path.join(
                "run_logs", "_".join([str(self.id), self.model_instance.config.model_type, self.dataset_manager.name])
            )

    def run(self):
        set_seed(self._train_run_args.seed)
        # Load dataset
        self.dataset_manager.load()
        # Init notifier
        if self.notifier and not self.notifier.title:
            name = ["#id" + str(self.id)]
            if self.model_instance.model_name:
                name.append(self.model_instance.model_name)
            if self.dataset_manager.name:
                name.append(self.dataset_manager.name)
            self.notifier.title = ", ".join(name)

        # Configure and run training
        if self._train_run_args:
            self._auto_fill_dirs(self._train_run_args)
            runner = Runner(
                self.model_instance,
                self._train_run_args,
                self.dataset_manager,
                do_save_full_model=not self.model_args.train_adapter,
                do_save_adapters=self.model_args.train_adapter,
            )
            if self.notifier:
                self.notifier.notify_start(message="Training setup:", **self._train_run_args.to_sanitized_dict())
            try:
                step, loss, best_score = runner.train(
                    self.model_args.model_name_or_path if os.path.isdir(self.model_args.model_name_or_path) else None
                )
                runner.save_model()
            except Exception as ex:
                if self.notifier:
                    self.notifier.notify_error(str(ex))
                raise ex
            # if no evaluation is done, we're at the end here
            if not self._eval_run_args and self.notifier:
                self.notifier.notify_end(message="Training results:", step=step, loss=loss, best_score=best_score)

        # Configure and run eval
        if self._eval_run_args:
            self._auto_fill_dirs(self._eval_run_args)
            runner = Runner(
                self.model_instance,
                self._eval_run_args,
                self.dataset_manager,
                do_save_full_model=not self.model_args.train_adapter,
                do_save_adapters=self.model_args.train_adapter,
            )
            try:
                results = runner.evaluate()
                output_eval_file = os.path.join(
                    self._eval_run_args.output_dir, f"eval_results_{self.dataset_manager.name}.txt"
                )
            except Exception as ex:
                if self.notifier:
                    self.notifier.notify_error(str(ex))
                raise ex
            with open(output_eval_file, "w") as f:
                logger.info("***** Eval results {} *****".format(self.dataset_manager.name))
                for key, value in results.items():
                    logger.info("  %s = %s", key, value)
                    f.write("%s = %s\n" % (key, value))
            if self.notifier:
                self.notifier.notify_end(message="Evaluation results:", **results)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Simple tool to setup Transformers training runs.")
    parser.add_argument("config", type=str, help="Path to the json file containing the full training setup.")
    parser.add_argument("--id", type=int, default=0, help="ID of this run.")
    args = parser.parse_args()

    # Load and run
    setup = Setup(id=args.id)
    setup.load_from_file(args.config)
    setup.run()


if __name__ == "__main__":
    main()
