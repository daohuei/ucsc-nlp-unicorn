from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader

from helper import print_stage
from config import BATCH_SIZE

print_stage("Loading Data")


def load_emotion_dataset():
    return load_dataset("emotion")


dataset = load_emotion_dataset()

train_set = dataset["train"]
# dev_set = dataset["validation"]
dev_set = Dataset.from_dict(dataset["validation"][:5])
test_set = dataset["test"]
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


def prepare_tokenized_dataset(prefix_prompt="", suffix_prompt=""):
    def tokenize(examples):
        prompted_text = [
            f"{prefix_prompt} {text} {suffix_prompt}"
            for text in examples["text"]
        ]
        tokenized_inputs = tokenizer(
            prompted_text, truncation=True, padding=True
        )
        tokenized_inputs["labels"] = examples["label"]
        return tokenized_inputs

    return tokenize


def get_data_loader(split="train", prefix_prompt="", suffix_prompt=""):
    assert split in ["train", "dev", "test"]

    dataset_map = {
        "train": train_set,
        "dev": dev_set,
        "test": test_set,
    }

    tokenized_set = dataset_map[split].map(
        prepare_tokenized_dataset(
            prefix_prompt=prefix_prompt, suffix_prompt=suffix_prompt
        ),
        batched=True,
        batch_size=BATCH_SIZE,
    )
    tokenized_set = tokenized_set.remove_columns(["text", "label"])
    data_loader = DataLoader(
        tokenized_set, batch_size=BATCH_SIZE, collate_fn=data_collator
    )
    return data_loader


if __name__ == "__main__":
    # print(dataset["validation"][:5])
    dev_loader = get_data_loader("dev", prompt="emotion: [MASK]")
    for batch in dev_loader:
        print(batch)
        break
