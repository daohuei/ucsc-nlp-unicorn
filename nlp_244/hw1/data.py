from collections import Counter, OrderedDict
from lib2to3.pgen2 import token

import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

from constant import DATA_PATH, SEED, TASK, MODEL, BATCH_SIZE
from vocab import Vocab

np.random.seed(SEED)


def remove_punct(utterance):
    result = (
        utterance.replace("-", "")
        .replace("'s", "")
        .replace(":", " ")
        .replace("'d", "")
        .replace(".", "")
        .replace(" a ", " ")
        .replace("'m", "")
        .replace("'t", "t")
    )
    return result


def load_data():
    train_df = pd.read_excel(DATA_PATH["train"])
    test_df = pd.read_excel(DATA_PATH["test"])

    # fix typo: I-movie => I_movie
    train_df["IOB Slot tags"] = train_df["IOB Slot tags"].apply(
        lambda tags: tags.replace("I-movie", "I_movie")
    )

    train_intent_df, val_intent_df = split_train_val_intent(train_df)
    train_slot_df, val_slot_df = split_train_val_slot(train_df)

    balance_intent_df = balance_intent_data(train_intent_df)
    balance_slot_df = balance_slot_data(train_slot_df)

    balance_intent_df["Core Relations"] = balance_intent_df[
        "Core Relations"
    ].apply(lambda text: text.split())
    val_intent_df["Core Relations"] = val_intent_df["Core Relations"].apply(
        lambda text: text.split()
    )
    balance_slot_df["IOB Slot tags"] = balance_slot_df["IOB Slot tags"].apply(
        lambda text: text.split()
    )
    val_slot_df["IOB Slot tags"] = val_slot_df["IOB Slot tags"].apply(
        lambda text: text.split()
    )

    return {
        "intent": {
            "train": balance_intent_df,
            "dev": val_intent_df,
            "test": test_df,
        },
        "slot": {
            "train": balance_slot_df,
            "dev": val_slot_df,
            "test": test_df,
        },
    }


def split_train_val_intent(df):
    """Split train_set to train/val set for Intent Detection

    Args:
        df (pandas.DataFrame): must contain the key of ["utterances", "Core Relations"]

    Returns:
        (pandas.DataFrame, pandas.DataFrame): return the train set df and val set df
    """
    train_intent_df = (
        df[["utterances", "Core Relations"]]
        .copy()
        .dropna(subset=["Core Relations"])
    )
    val_intent_df = train_intent_df.sample(0)
    for intent in train_intent_df["Core Relations"].unique():
        # sample 30% from train data for each intent to keep them balanced
        intent_idx = train_intent_df["Core Relations"] == intent
        intent_size = len(train_intent_df[intent_idx])
        val_intent_size = int(intent_size * 0.3)
        val_intent_df = pd.concat(
            [
                val_intent_df,
                train_intent_df[intent_idx].sample(val_intent_size),
            ]
        )
    train_intent_df = train_intent_df.drop(val_intent_df.index)
    return train_intent_df, val_intent_df


def split_train_val_slot(df):
    """Split train_set to train/val set for Slot Tagging

    Args:
        df (pandas.DataFrame): must contain the key of ["utterances", "IOB Slot tags"]

    Returns:
        (pandas.DataFrame, pandas.DataFrame): return the train set df and val set df
    """

    train_slot_df = (
        df[["utterances", "IOB Slot tags"]]
        .copy()
        .dropna(subset=["IOB Slot tags"])
    )

    # generate the unique set of the tags that the tag sequence contains
    train_slot_df["slot_set"] = train_slot_df["IOB Slot tags"].apply(
        lambda slot_text: " ".join(sorted(list(set(slot_text.split()))))
    )

    val_slot_df = train_slot_df.sample(0)

    # split it according to the unique slot set to keep tags balance in both set
    for slot_set_type in train_slot_df["slot_set"].unique():
        slot_idx = train_slot_df["slot_set"] == slot_set_type
        slot_size = len(train_slot_df[slot_idx])
        val_slot_size = int(slot_size * 0.3)
        val_slot_df = pd.concat(
            [val_slot_df, train_slot_df[slot_idx].sample(val_slot_size)]
        )
    train_slot_df = train_slot_df.drop(val_slot_df.index)
    return train_slot_df, val_slot_df


def build_counter(df, key):
    """Build the distribution counter for labels

    Args:
        df (pandas.DataFrame): input DataFrame, must contain "Core Relations" or "slot_set" column
        key (str): either "Core Relations" or "slot_set"

    Returns:
        OrderedDict: Ordered counter for labels
    """
    counter = Counter()
    if key == "Core Relations":
        for label in df[key].dropna():
            labels = label.split()
            counter.update(labels)
    else:
        counter = Counter(df[key])
    return OrderedDict(counter.most_common())


def balance_intent_data(df):
    """Balance the data for intent detection task

    Args:
        df (pandas.DataFrame): data to be balance, should contain "Core Relations" column

    Returns:
        pandas.DataFrame: balanced data
    """
    relation_counter = build_counter(df, "Core Relations")

    # augment each low resource label to average count
    avg_count = int(
        sum(relation_counter.values()) / len(relation_counter.values())
    )
    sample_df = df.sample(0)

    for k, v in relation_counter.items():
        # only augment the low resource label
        if v >= avg_count:
            continue
        # to be sample amount
        sample_count = avg_count - v

        idx_of_label_k = df["Core Relations"].apply(lambda label: k in label)

        # if sample amount if larger, then sample all the value until it exceed the sample count
        while sample_count > relation_counter[k]:
            temp_df = df[idx_of_label_k].sample(relation_counter[k])
            sample_df = pd.concat([sample_df, temp_df])
            sample_count -= relation_counter[k]

        sample_df = pd.concat(
            [sample_df, df[idx_of_label_k].sample(sample_count)]
        )

    balance_df = pd.concat([df.copy(), sample_df])

    return balance_df


def balance_slot_data(df):
    """Balance the data for slot tagging task

    Args:
        df (pandas.DataFrame): data to be balance, should contain "slot_set" column

    Returns:
        pandas.DataFrame: balanced data
    """
    slot_counter = build_counter(df, "slot_set")

    # augment each low resource label to average count
    avg_count = int(sum(slot_counter.values()) / len(slot_counter.values()))
    sample_df = df.sample(0)

    for k, v in slot_counter.items():
        # only augment the low resource label
        if v >= avg_count:
            continue
        # ignore the tag sequence only contains "O"
        if k == "O":
            continue
        # to be sample amount
        sample_count = avg_count - v

        idx_of_label_k = df["slot_set"].apply(lambda label: k == label)

        # if sample amount if larger, then sample all the value until it exceed the sample count
        while sample_count > slot_counter[k]:
            temp_df = df[idx_of_label_k].sample(slot_counter[k])
            sample_df = pd.concat([sample_df, temp_df])
            sample_count -= slot_counter[k]

        sample_df = pd.concat(
            [sample_df, df[idx_of_label_k].sample(sample_count)]
        )
    balance_df = pd.concat([df.copy(), sample_df])

    return balance_df


def prepare_dataset(df_dict, label_key):
    train_df = df_dict["train"]
    dev_df = df_dict["dev"]
    test_df = df_dict["test"]

    train_movie_dataset = Dataset.from_pandas(
        train_df[["utterances", label_key]],
        preserve_index=False,
    )

    dev_movie_dataset = Dataset.from_pandas(
        dev_df[["utterances", label_key]],
        preserve_index=False,
    )
    test_movie_dataset = Dataset.from_pandas(
        test_df[["utterances"]],
        preserve_index=False,
    )
    return (train_movie_dataset, dev_movie_dataset, test_movie_dataset)


df_dict = load_data()
slot_vocab = Vocab(
    df_dict["slot"]["train"]["IOB Slot tags"].tolist(), base_map={}
)

train_set, dev_set, test_set = None, None, None
pretrained_tokenizer = None
if TASK == "slot":
    train_set, dev_set, test_set = prepare_dataset(
        df_dict["slot"], "IOB Slot tags"
    )
elif TASK == "intent":
    train_set, dev_set, test_set = prepare_dataset(
        df_dict["intent"], "Core Relations"
    )

if MODEL == "distil_bert":
    pretrained_tokenizer = AutoTokenizer.from_pretrained(
        "distilbert-base-uncased"
    )

data_collator = DataCollatorWithPadding(tokenizer=pretrained_tokenizer)


def prepare_slot_dataset(dataset, tokenizer, split):
    tokenized_set = dataset.map(
        slot_tokenize(tokenizer, slot_vocab, split), batched=True
    )
    tokenized_set = tokenized_set.remove_columns(
        ["utterances", "IOB Slot tags"]
    )
    tokenized_set.set_format("torch")
    return tokenized_set


def slot_tokenize(tokenizer, slot_vocab, split):
    """Tokenization and aligning labels for slot tagging task"""

    def tokenize(examples):

        # remove punct symbol in utterance, since the tokenizer not consider punct as the sub word
        examples["utterances"] = [
            remove_punct(text) for text in examples["utterances"]
        ]

        tokenized_inputs = tokenizer(
            examples["utterances"], truncation=True, padding=True
        )

        # if it is the test set, we do not need to do label alignment on word piece
        if split == "test":
            return tokenized_inputs

        slot_indexes = [
            [slot_vocab.lookup_index(slot) for slot in slot_seq]
            for slot_seq in examples["IOB Slot tags"]
        ]

        # aligning the label to wordpiece tokens
        labels = []
        for i, label in enumerate(slot_indexes):
            word_ids = tokenized_inputs.word_ids(
                batch_index=i
            )  # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set the special tokens to -100.
                if word_idx is None:
                    label_ids.append(-100)
                elif (
                    word_idx != previous_word_idx
                ):  # Only label the first token of a given word.
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    return tokenize


def get_data_loader(split="train"):
    dataset_map = {
        "train": train_set,
        "dev": dev_set,
        "test": test_set,
    }

    tokenized_set = prepare_slot_dataset(
        dataset_map[split], pretrained_tokenizer, split
    )
    data_loader = DataLoader(
        tokenized_set, batch_size=BATCH_SIZE, collate_fn=data_collator
    )
    return data_loader


if __name__ == "__main__":
    for batch in get_data_loader():
        break
    print({k: v.shape for k, v in batch.items()})
