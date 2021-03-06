from collections import Counter

import pandas as pd
import spacy
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from helper import print_stage, read_files
from constant import (
    SRC_FILENAME,
    TGT_FILENAME,
    START,
    STOP,
    PADDING,
    UNK_TOKEN,
    DEVICE,
    TRUNCATE_SIZE,
    MAX_VOCAB_SIZE,
    LEAST_FREQ,
)

spacy_en = spacy.load("en_core_web_sm")


class CNNDailyMailDataset(Dataset):
    def __init__(self, data):
        self.X = []
        self.y = []
        self.raw_y = []
        for text, summary in data:
            if len(text) > 0:
                self.X.append(text)
                self.y.append(summary)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index], index


class Vocab:
    def __init__(self, tokens, base_map={}, max_size=None, least_freq=0):
        self.token2idx = base_map
        # count the word/token/tags frequency
        self.freq = Counter(
            [token for sequence in tokens for token in sequence]
        )

        vocab_size = 0
        # store the token start from higher frequency
        for word, count in sorted(
            self.freq.items(), key=lambda item: item[1], reverse=True
        ):
            if count < least_freq:
                break
            # if vocab size is larger than max size, stop inserting words into vocab
            if max_size is not None and vocab_size > max_size:
                break
            self.insert(word)
            vocab_size += 1

        self.idx2token = reverse_map(self.token2idx)

    def insert(self, token):
        if token in self.token2idx.keys():
            return
        self.token2idx[token] = len(self.token2idx)

    def lookup_index(self, word):
        if word not in self.token2idx.keys():
            word = UNK_TOKEN
        return self.token2idx[word]

    def lookup_token(self, idx):
        return self.idx2token[idx]

    def __len__(self):
        return len(self.token2idx)

    def __repr__(self):
        return str(self.token2idx)


def reverse_map(_map):
    reversed_map = {}
    for key, val in _map.items():
        reversed_map[val] = key
    return reversed_map


def add_special_tokens(tokens):
    return [START] + tokens.split() + [STOP]


def build_data_points(data_df):
    return data_df.apply(
        lambda row: (
            add_special_tokens(row["text"]),
            add_special_tokens(row["summary"]),
        ),
        axis=1,
    )


def load_data(split, line_constraint=None, truncate=False):
    inputs = read_files(
        SRC_FILENAME[f"{split}_truncate"]
        if truncate and split == "train"
        else SRC_FILENAME[split],
        line_constraint,
    )
    outputs = read_files(TGT_FILENAME[split], line_constraint)
    df = pd.DataFrame({"text": inputs, "summary": outputs})
    max_len_summary = max(
        outputs, key=lambda summary: len(summary.split())
    ).split()
    max_len = len(max_len_summary)
    data_point_df = build_data_points(df)
    return data_point_df, max_len


print_stage("Loading Training Data")
train_data, _ = load_data("train")[:]
print_stage("Loading Dev Data")
dev_data, dev_summary_max_len = load_data("dev")[:]
print_stage("Loading Test Data")
test_data, test_summary_max_len = load_data("test")[:]

summary_max_len = max(dev_summary_max_len, test_summary_max_len)

train_set = CNNDailyMailDataset(train_data)
dev_set = CNNDailyMailDataset(dev_data)
test_set = CNNDailyMailDataset(test_data)

print_stage("Building Vocab")
word_vocab = Vocab(
    train_set.X + train_set.y,
    base_map={PADDING: 0, UNK_TOKEN: 1},
    max_size=MAX_VOCAB_SIZE,
    least_freq=LEAST_FREQ,
)

print("Vocab Length=", len(word_vocab))

# text preprocess pipeline
def text_pipeline(sentence, truncate_size=TRUNCATE_SIZE):
    if len(sentence) > truncate_size:
        sentence = sentence[:truncate_size]
    return [word_vocab.lookup_index(token) for token in sentence]


# preprocess batch data before loading each batch
def collate_batch(batch):
    summary_list, text_list, index_list = [], [], []
    for _text, _summary, _index in batch:
        text_list.append(torch.tensor(text_pipeline(_text), dtype=torch.long,))
        summary_list.append(
            torch.tensor(
                text_pipeline(_summary, truncate_size=len(_summary)),
                dtype=torch.long,
            )
        )
        index_list.append(torch.tensor(_index, dtype=torch.long))

    len_list = torch.tensor(list(map(len, text_list)), dtype=torch.long)
    text_list = pad_sequence(text_list, batch_first=True, padding_value=0)
    summary_list = pad_sequence(
        summary_list, batch_first=True, padding_value=0
    )
    index_list = torch.tensor(index_list, dtype=torch.long)

    # sort the batch according to the sequence length in the descending order
    len_list, perm_idx = len_list.sort(0, descending=True)
    text_list = text_list[perm_idx]
    summary_list = summary_list[perm_idx]
    index_list = index_list[perm_idx]

    return text_list.to(DEVICE), summary_list.to(DEVICE), len_list, index_list


def get_data_loader(batch_size: int = 1, set_name="train"):
    assert set_name in ["train", "dev", "test"]
    # use data loader for batching data
    if set_name == "train":
        return DataLoader(
            dataset=train_set,
            batch_size=batch_size,
            collate_fn=collate_batch,
            shuffle=False,
        )
    elif set_name == "dev":
        return DataLoader(
            dataset=dev_set,
            batch_size=batch_size,
            collate_fn=collate_batch,
            shuffle=False,
        )
    else:
        return DataLoader(
            dataset=test_set,
            batch_size=batch_size,
            collate_fn=collate_batch,
            shuffle=False,
        )
