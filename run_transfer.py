import argparse
import json
import os

from itrain import DATASET_MANAGER_CLASSES, DatasetArguments, ModelArguments, RunArguments, Setup


TRANSFER_OUTPUT_DIR="transfer_output"
FUSION_OUTPUT_DIR="fusion_output"
RUN_CONFIGS="run_configs"

def _get_dataset_config(config_name, train_size=-1):
    # init setup
    with open(os.path.join(RUN_CONFIGS, config_name+".json"), "r", encoding="utf-8") as f:
        config = json.load(f)
    # dataset manager
    config["dataset"]["train_subset_size"] = train_size
    dataset_args = DatasetArguments(**config["dataset"])
    dataset_manager = DATASET_MANAGER_CLASSES[dataset_args.dataset_name](dataset_args)
    return dataset_manager, config


def _restore_path(adapter_map, task_name, manager):
    template = adapter_map["source_path_format"]
    run_id = adapter_map["adapters"][task_name]
    # HACK: the actual path to the adapter may have different names
    path = os.path.expanduser(template.format(task_name, run_id, manager.name))
    if not os.path.exists(path):
        path = os.path.expanduser(template.format(manager.name, run_id, manager.name))
    return path


def run_seq_finetuning(args):
    # init setup
    dataset_manager, config = _get_dataset_config(args["target_task"], train_size=args["train_size"])
    if args["train_size"] > 0:
        target_task_name = args["target_task"] + "_n" + str(args["train_size"])
    else:
        target_task_name = args["target_task"]
    output_base = os.path.join(TRANSFER_OUTPUT_DIR, "to_" + target_task_name)
    # patch model/ training args
    config["training"]["learning_rate"] = args["learning_rate"] or 1e-4
    config["training"]["num_train_epochs"] = args["num_train_epochs"]

    # load results if existing
    final_results_file = os.path.join(output_base, "eval_results.json")
    if os.path.exists(final_results_file):
        with open(final_results_file, "r") as f:
            results = json.load(f)
    else:
        results = {}

    # iterate over adapters for fusion
    with open(args["trained_adapter_map"], "r") as f:
        trained_adapter_map = json.load(f)
    for task_name in trained_adapter_map["from"]:
        print(f"*** Running transfer from {task_name} to {target_task_name} ***")
        output_dir = os.path.join(output_base, task_name)
        # skip this iteration if no overwrites requested & existing
        if not args["overwrite_output"] and os.path.exists(output_dir):
            print(f"Skipping task {task_name} as it already exists.")
            continue

        pre_training_dataset_manager, _ = _get_dataset_config(task_name)
        setup = Setup(id=args["id"])
        setup.dataset(dataset_manager)
        config["training"]["output_dir"] = output_dir
        setup.training(RunArguments(**config["training"]))
        if isinstance(config["evaluation"], str):
            setup.evaluation(split=config["evaluation"])
        else:
            setup.evaluation()
        setup.notify(config["notify"])
        setup._config_name = "transfer_" + task_name + "_to_" + target_task_name
        # setup model
        config["model"]["load_adapters"] = {
            dataset_manager.name: _restore_path(trained_adapter_map, task_name, pre_training_dataset_manager)
        }
        setup.model(ModelArguments(**config["model"]))
        # start!
        run_results = setup.run(restarts=args["restarts"])
        results[task_name] = run_results

    # save results
    with open(final_results_file, "w") as f:
        json.dump(results, f)


# def run_fusion_seq(args):
#     results = {}
#     # init setup
#     dataset_manager, config = _get_dataset_config(args["target_task"], train_size=args["train_size"])
#     output_base = os.path.join(FUSION_OUTPUT_DIR, "to_" + args["target_task"] + "_v2")  # TODO
#     # patch model/ training args
#     config["model"]["train_adapter"] = False
#     config["training"]["learning_rate"] = args["learning_rate"] or 5e-5
#     config["training"]["num_train_epochs"] = args["num_train_epochs"]
#     # iterate over adapters for fusion
#     with open(args["trained_adapter_map"], "r") as f:
#         trained_adapter_map = json.load(f)
#     for task_name in trained_adapter_map["from"]:
#         fusion_dataset_manager, _ = _get_dataset_config(task_name)
#         setup = Setup(id=args["id"])
#         setup.dataset(dataset_manager)
#         output_dir = os.path.join(output_base, task_name)
#         config["training"]["output_dir"] = output_dir
#         setup.training(RunArguments(**config["training"]))
#         if isinstance(config["evaluation"], str):
#             setup.evaluation(split=config["evaluation"])
#         else:
#             setup.evaluation()
#         setup.notify(config["notify"])
#         setup._config_name = "fusion_" + args["target_task"] + "_" + task_name
#         # setup model
#         config["model"]["load_adapters"] = {
#             dataset_manager.name: _restore_path(trained_adapter_map, args["target_task"], dataset_manager),
#             fusion_dataset_manager.name: _restore_path(trained_adapter_map, task_name, fusion_dataset_manager)
#         }
#         config["model"]["train_adapter_fusion"] = ",".join([dataset_manager.name, fusion_dataset_manager.name])
#         setup.model(ModelArguments(**config["model"]))
#         # start!
#         run_results = setup.run(restarts=args["restarts"])
#         results[task_name] = run_results
#     # save results
#     with open(os.path.join(output_base, "eval_results.json"), "w") as f:
#         json.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("target_task", type=str, help="Name of the target task training setup.")
    parser.add_argument("--id", type=int, default=0, help="ID of this run.")
    parser.add_argument("--trained_adapter_map", type=str, required=True)
    parser.add_argument("--overwrite_output", action="store_true", default=False)
    parser.add_argument("--fusion", action="store_true", default=False)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--num_train_epochs", type=int, default=15)
    parser.add_argument("--train_size", type=int, default=-1)
    parser.add_argument("--restarts", type=int, default=None)
    args = vars(parser.parse_args())

    fusion = args.pop("fusion")

    if fusion:
        raise ValueError()
        # run_fusion_seq(args)
    else:
        run_seq_finetuning(args)
