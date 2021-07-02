from itrain import DatasetArguments, ModelArguments, RunArguments, Setup


setup = Setup()

# Set up dataset
setup.dataset(DatasetArguments(
    dataset_name="glue",
    task_name="rte",
    max_seq_length=290,
))

# Set up model
setup.model(ModelArguments(
    model_name_or_path="roberta-base",
    use_fast_tokenizer=True,
    # load_adapters=[
    # ],
    # train_adapter_fusion="mnli,qnli",
    train_adapter=True,
    adapter_config="pfeiffer",
))

# Set up training
setup.training(RunArguments(
    learning_rate=1e-4,
    batch_size=32,
    num_train_epochs=15,
    patience=4,
    patience_metric="eval_accuracy"
))

# Evaluate
setup.evaluation()

# Training notification
# setup.notify("telegram")

results = setup.run(restarts=1)
print(results)
