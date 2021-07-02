import argparse
import json
import os

from config_utils import get_dataset_config, restore_path
from itrain import ModelArguments, RunArguments, Setup


TRANSFER_OUTPUT_DIR = os.path.expanduser(os.getenv("RUN_TRANSFER_OUTPUT_DIR", "transfer_output"))
FUSION_OUTPUT_DIR = "fusion_output"


def run_seq_finetuning(args):
    # init setup
    dataset_manager, config = get_dataset_config(args["target_task"], train_size=args["train_size"])
    if args["train_size"] > 0:
        target_task_name = args["target_task"] + "_n" + str(args["train_size"])
    else:
        target_task_name = args["target_task"]
    output_base = os.path.join(TRANSFER_OUTPUT_DIR, "to_" + target_task_name)
    # patch model/ training args
    if args["full_model"]:
        config["model"]["train_adapter"] = False
        config["training"]["patience"] = 0
        default_lr = 3e-5
        default_epochs = 3
    else:
        default_lr = 1e-4
        default_epochs = 15
    config["training"]["learning_rate"] = args["learning_rate"] or default_lr
    config["training"]["num_train_epochs"] = args["num_train_epochs"] or default_epochs

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
        if args["overwrite_mode"] == 0 and os.path.exists(output_dir):
            print(f"Skipping task {task_name} as it already exists.")
            continue
        # also skip if appending & max runs reached
        elif (
            args["overwrite_mode"] == 1
            and args["max_restarts"] is not None
            and task_name in results
            and len(results[task_name]["seeds"]) >= args["max_restarts"]
        ):
            print(f"Skipping task {task_name} as it already reached {args['max_restarts']} runs.")
            continue

        pre_training_dataset_manager, _ = get_dataset_config(task_name)
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
        if args["model_name_or_path"]:
            config["model"]["model_name_or_path"] = args["model_name_or_path"]
        if args["full_model"]:
            if not config["model"].get("tokenizer_name", None):
                # HACK in case the tokenizer is not saved with the model
                config["model"]["tokenizer_name"] = config["model"]["model_name_or_path"]
            config["model"]["model_name_or_path"] = restore_path(
                trained_adapter_map, task_name, pre_training_dataset_manager
            )
            config["model"]["drop_model_head"] = True
        else:
            config["model"]["load_adapters"] = {
                dataset_manager.name: restore_path(trained_adapter_map, task_name, pre_training_dataset_manager)
            }
        setup.model(ModelArguments(**config["model"]))
        # start!
        if task_name in results and args["overwrite_mode"] == 1:
            # append to existing
            run_results = setup.run(restarts=args["restarts"], first_run_index=len(results[task_name]["seeds"]))
            for k, v in run_results.items():
                results[task_name][k] += v
        else:
            run_results = setup.run(restarts=args["restarts"])
            results[task_name] = run_results
        del setup

        # save results after every iteration
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
    parser.add_argument(
        "--overwrite_mode", type=int, choices=[0, 1, 2], default=0, help="0: no overwrite; 1: append; 2: overwrite"
    )
    parser.add_argument("--fusion", action="store_true", default=False)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--num_train_epochs", type=int, default=None)
    parser.add_argument("--train_size", type=int, default=-1)
    parser.add_argument("--restarts", type=int, default=None)
    parser.add_argument("--max_restarts", type=int, default=None)
    parser.add_argument("--full_model", action="store_true")
    parser.add_argument("--model_name_or_path", default=None)
    args = vars(parser.parse_args())

    fusion = args.pop("fusion")

    if fusion:
        raise ValueError()
        # run_fusion_seq(args)
    else:
        run_seq_finetuning(args)
