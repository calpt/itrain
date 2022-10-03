# itrain

**⚠ WARNING: This software is developed for personal research experiments and subject to frequent changes/ fixes.** 

> Ready-to-run scripts for Transformers and Adapters on >50 NLP tasks.

This repository contains `itrain`, a small library that provides a simple interface for configuring training runs of **[Transformers](https://github.com/huggingface/transformers)** and **[Adapters](https://github.com/Adapter-Hub/adapter-transformers)** across [a wide range of NLP tasks](run_configs).

The code is based on the research code of the paper ["What to Pre-Train on? Efficient Intermediate Task Selection"](https://arxiv.org/pdf/2104.08247), which can be found [here](https://github.com/adapter-hub/efficient-task-transfer).

## Feature Overview

The `itrain` package provides:
- easy downloading and preprocessing datasets via HuggingFace datasets
- integration of a wide range of standard NLP tasks ([list](run_configs))
- training run setup & configuration via Python or command-line
- automatic checkpointing, WandB logging, resuming & random restarts for score distributions
- automatic notification on training start and results via mail or Telegram chat

## Setup & Requirements

Before getting started with this repository, make sure to have a recent Python version (> 3.6) and PyTorch ([see here](https://pytorch.org/get-started/locally/)) set up (ideally in a virtual environment such as conda).

All additional requirements together with the `itrain` package can be installed by cloning this repository and then installing from source:
```bash
git clone https://github.com/calpt/itrain.git
cd itrain
pip install -e .
```

## How To Use

### Command-line

`itrain` can be invoked from the command line by passing a run configuration file in YAML or JSON format.
Example configurations for all currently supported tasks can be found in the [run_configs](run_configs) folder.
All supported configuration keys are defined in [arguments.py](itrain/arguments.py).

Running a setup from the command line can look like this:
```bash
itrain run --id 42 run_configs/sst2.json
```
This will train an adapter on the SST-2 task using `robert-base` as the base model (as specified in the config file).

Besides modifying configuration keys directly in the json file, they can be overriden using command line parameters.
E.g., we can modify the previous training run to fully fine-tune a `bert-base-uncased` model:
```bash
itrain run --id 42 \
    --model_name_or_path bert-base-uncased \
    --train_adapter false \
    --learning_rate 3e-5 \
    --num_train_epochs 3 \
    --patience 0 \
    run_configs/sst2.json
```

### Python script

Alternatively, training setups can be configured directly in Python by using the `Setup` class of `itrain`. An example for this is given in [example.py](example.py).

## Credits

- **[huggingface/transformers](https://github.com/huggingface/transformers)** for the Transformers implementations, the trainer class and the training scripts on which this repository is based
- **[huggingface/datasets](https://github.com/huggingface/datasets)** for dataset downloading and preprocessing
- **[Adapter-Hub/adapter-transformers](https://github.com/Adapter-Hub/adapter-transformers)** for the adapter implementation
