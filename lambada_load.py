from datasets import load_dataset

dataset = load_dataset("lambada")

# Accessing train, test, and validation splits
train_dataset = dataset["train"]
test_dataset = dataset["test"]
validation_dataset = dataset["validation"]