import argparse
import dataclasses
import os

from .arguments import DatasetArguments, ModelArguments, RunArguments
from .datasets import CacheMode
from .itrain import SETUP_OUTPUT_FILE, STATE_OUTPUT_FILE, Setup


def main():

    parser = argparse.ArgumentParser(description="Simple tool to setup Transformers training runs.")
    subparsers = parser.add_subparsers(dest="command")

    # "run" command
    parser_run = subparsers.add_parser("run", help="Run setup.")
    parser_run.add_argument("config", type=str, help="Path to the json file containing the full training setup.")
    parser_run.add_argument("--id", type=int, default=0, help="ID of this run.")
    parser_run.add_argument(
        "--preprocess_only", action="store_true", default=False, help="Only run dataset preprocessing."
    )
    restarts_group = parser_run.add_mutually_exclusive_group()
    restarts_group.add_argument("--seeds", type=lambda s: [int(item) for item in s.split(",")])
    restarts_group.add_argument("--restarts", type=int, default=None)
    # add arguments from dataclasses
    for dtype in (DatasetArguments, ModelArguments, RunArguments):
        for field in dataclasses.fields(dtype):
            field_name = f"--{field.name}"
            kwargs = field.metadata.copy()
            kwargs["type"] = field.type
            kwargs["default"] = None
            parser_run.add_argument(field_name, **kwargs)

    # "resume" command
    parser_resume = subparsers.add_parser("resume", help="Resume setup.")
    parser_resume.add_argument("directory", type=str, help="Path to the directory holding a previously run setup.")
    restarts_group_2 = parser_resume.add_mutually_exclusive_group()
    restarts_group_2.add_argument("--add_seeds", type=lambda s: [int(item) for item in s.split(",")])
    restarts_group_2.add_argument("--add_restarts", type=int, default=None)

    args = vars(parser.parse_args())

    # Load and run
    if args["command"] == "run":
        setup = Setup.from_file(args["config"], overrides=args)
        setup.id = args.pop("id")
        if args.pop("preprocess_only"):
            setup._setup_tokenizer()
            setup.dataset_manager.load_and_preprocess(CacheMode.USE_DATASET_NEW_FEATURES)
        else:
            setup.run(restarts=args.pop("seeds") or args.pop("restarts"))
    elif args["command"] == "resume":
        setup = Setup.from_file(os.path.join(args["directory"], SETUP_OUTPUT_FILE))
        setup.resume(
            os.path.join(args["directory"], STATE_OUTPUT_FILE),
            add_restarts=args.pop("add_seeds") or args.pop("add_restarts"),
        )


if __name__ == "__main__":
    main()
