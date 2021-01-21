import argparse
import json
import os

from itrain import DATASET_MANAGER_CLASSES, DatasetArguments, ModelArguments, RunArguments, Setup


OUTPUT_DIR="limited_output"
RUN_CONFIGS="run_configs"


def run_limited_training(args):
    results = {}
    # load training config
    with open(os.path.join(RUN_CONFIGS, args["target_task"]+".json"), "r", encoding="utf-8") as f:
        config = json.load(f)
    output_base = os.path.join(OUTPUT_DIR, args["target_task"])
    if not os.path.exists(output_base):
        os.makedirs(output_base)
    # patch model/ training args
    config["training"]["learning_rate"] = args["learning_rate"]
    config["training"]["checkpoint_steps"] = 100
    for dataset_size in args["train_sizes"]:
        print(f"*** Running {args['target_task']} with size = {dataset_size} ***")
        output_dir = os.path.join(output_base, str(dataset_size))
        # skip this iteration if no overwrites requested & existing
        if not args["overwrite_output"] and os.path.exists(output_dir):
            print(f"Skipping size {dataset_size} as it already exists.")
            continue
        # create dataset
        config["dataset"]["train_subset_size"] = dataset_size
        dataset_args = DatasetArguments(**config["dataset"])
        dataset_manager = DATASET_MANAGER_CLASSES[dataset_args.dataset_name](dataset_args)
        # create run setup
        setup = Setup(id=f"{args['target_task']}_n{dataset_size}")
        setup.dataset(dataset_manager)
        config["training"]["output_dir"] = output_dir
        setup.training(RunArguments(**config["training"]))
        if isinstance(config["evaluation"], str):
            setup.evaluation(split=config["evaluation"])
        else:
            setup.evaluation()
        setup.notify(config["notify"])
        # setup model
        setup.model(ModelArguments(**config["model"]))
        # start!
        run_results = setup.run(restarts=args["restarts"])
        results[str(dataset_size)] = run_results
        # stop training if the current dataset size exceeds the total size
        if len(dataset_manager.train_split) <= dataset_size:
            print(f"Dataset size exceeded at n = {dataset_size}, stopping.")
            break
        # clean up
        del dataset_manager
        del setup
    # save results
    with open(os.path.join(output_base, "eval_results.json"), "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("target_task", type=str, help="Name of the target task training setup.")
    parser.add_argument("--overwrite_output", action="store_true", default=False)
    parser.add_argument("--train_sizes", type=lambda s: [int(item) for item in s.split(",")])
    parser.add_argument("--restarts", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    args = vars(parser.parse_args())

    run_limited_training(args)
