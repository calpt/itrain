# itrain

Training code for ["What to Pre-Train on? Efficient Intermediate Task Selection"](https://arxiv.org/abs/2104.08247).

## Setup

```bash
pip install -e .
```

## Tasks

See [supported tasks](run_configs) and [dataset managers](itrain/datasets).

## Run

### From Python
See [example.py](example.py)

### Adapter (RoBERTa)
```bash
itrain --id <run_id> run_configs/<task>.json
```

### Adapter (BERT)
```bash
itrain --id <run_id> \
    --model_name_or_path bert-base-uncased \
    run_configs/<task>.json
```

### Full fine-tuning (BERT)
```bash
itrain --id <run_id> \
    --model_name_or_path bert-base-uncased \
    --train_adapter false \
    --learning_rate 3e-5 \
    --num_train_epochs 3 \
    --patience 0 \
    run_configs/<task>.json
```
