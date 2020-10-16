from transformers import AdapterConfig, AdapterType, AutoConfig, AutoModelWithHeads, AutoTokenizer

from .arguments import ModelArguments
from .datasets import DatasetManager


def create_tokenizer(args: ModelArguments):
    return AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        use_fast=args.use_fast_tokenizer,
    )


def create_model(args: ModelArguments, manager: DatasetManager):
    config = AutoConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    model = AutoModelWithHeads.from_pretrained(args.model_name_or_path, config=config)
    model.add_prediction_head(manager.name, manager.get_prediction_head_config())

    # setup adapters
    if args.train_adapter:
        adapter_config = AdapterConfig.load(args.adapter_config)
        model.add_adapter(manager.name, AdapterType.text_task, config=adapter_config)
        model.train_adapter([manager.name])

    return model
