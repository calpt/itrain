import argparse
import json
import os

from itrain import DATASET_MANAGER_CLASSES, DatasetArguments, ModelArguments, RunArguments, Setup


FUSION_OUTPUT_DIR="fusion_output"
RUN_CONFIGS="run_configs"

def _get_dataset_config(config_name):
    # init setup
    with open(os.path.join(RUN_CONFIGS, config_name+".json"), "r", encoding="utf-8") as f:
        config = json.load(f)
    # dataset manager
    dataset_args = DatasetArguments(**config["dataset"])
    dataset_manager = DATASET_MANAGER_CLASSES[dataset_args.dataset_name](dataset_args)
    return dataset_manager, config


def _restore_path(adapter_map, task_name, manager):
    template = adapter_map["path_format"]
    run_id = adapter_map["adapters"][task_name]
    # HACK: the actual path to the adapter may have different names
    path = os.path.expanduser(template.format(task_name, run_id, manager.name))
    if not os.path.exists(path):
        path = os.path.expanduser(template.format(manager.name, run_id, manager.name))
    return path


def run_fusion_seq(args):
    results = {}
    # init setup
    dataset_manager, config = _get_dataset_config(args["target_task"])
    output_base = os.path.join(FUSION_OUTPUT_DIR, "to_" + args["target_task"])
    # patch model/ training args
    config["model"]["train_adapter"] = False
    config["training"]["learning_rate"] = args["learning_rate"]
    config["training"]["num_train_epochs"] = args["num_train_epochs"]
    # iterate over adapters for fusion
    with open(args["trained_adapter_map"], "r") as f:
        trained_adapter_map = json.load(f)
    for task_name in trained_adapter_map["from"]:
        fusion_dataset_manager, _ = _get_dataset_config(task_name)
        setup = Setup(id=args["id"])
        setup.dataset(dataset_manager)
        output_dir = os.path.join(output_base, task_name)
        config["training"]["output_dir"] = output_dir
        setup.training(RunArguments(**config["training"]))
        setup.evaluation()
        setup.notify(config["notify"])
        setup._config_name = "fusion_" + args["target_task"] + "_" + task_name
        # setup model
        config["model"]["load_adapters"] = [
            _restore_path(trained_adapter_map, args["target_task"], dataset_manager),
            _restore_path(trained_adapter_map, task_name, fusion_dataset_manager)
        ]
        config["model"]["train_adapter_fusion"] = ",".join([dataset_manager.name, fusion_dataset_manager.name])
        setup.model(ModelArguments(**config["model"]))
        # start!
        run_results = setup.run()
        results[task_name] = run_results
    # save results
    with open(os.path.join(output_base, "eval_results.json"), "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("target_task", type=str, help="Name of the target task training setup.")
    parser.add_argument("--id", type=int, default=0, help="ID of this run.")
    parser.add_argument("--trained_adapter_map", type=str, required=True)
    parser.add_argument("--overwrite_output", action="store_true", default=False)
    parser.add_argument("--seq", action="store_true", default=False)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_train_epochs", type=int, default=8)
    args = vars(parser.parse_args())

    seq = args.pop("seq")

    if not seq:
        raise ValueError()
    else:
        run_fusion_seq(args)
