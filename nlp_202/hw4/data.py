from collections import Counter
import random
from random import sample

import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

from constants import START_TAG, STOP_TAG, PADDING, UNK_TOKEN, DEVICE

random.seed(1)


class BioDataset(Dataset):
    def __init__(self, data):
        self.X = []
        self.y = []
        for sent, tag in data:
            if len(sent) > 0:
                self.X.append(sent)
                self.y.append(tag)

    # Must have
    def __len__(self):
        return len(self.y)

    # Must have
    def __getitem__(self, index):
        return self.X[index], self.y[index], index


class Vocab:
    def __init__(self, tokens, base_map={}, max_size=None):
        self.token2idx = base_map
        # count the word/token/tags frequency
        self.freq = Counter(
            [token for sequence in tokens for token in sequence]
        )

        vocab_size = 0
        # store the token start from higher frequency
        for word, _ in sorted(
            self.freq.items(), key=lambda item: item[1], reverse=True
        ):
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


def read_data(filename):
    """
    Reads the data into an array of dictionaries (a dictionary for each data point).
    :param filename: String
    :return: Array of dictionaries.  Each dictionary has the format returned by the make_data_point function.
    """
    data = []
    with open(filename, "r") as f:
        sent = []
        for line in f.readlines():
            if line.strip():
                sent.append(line)
            else:
                data.append(make_data_point(sent))
                sent = []
        data.append(make_data_point(sent))

    return data


def make_data_point(sent):
    """
        Creates a dictionary from String to an Array of Strings representing the data.  The dictionary items are:
        dic['tokens'] = Tokens padded with <START> and <STOP>
        dic['gold_tags'] = The gold tags padded with <START> and <STOP>
    :param sent: String.  The input (token tag) sequences
    :return: Dict from String to Array of Strings.
    """
    dic = {}
    sent = [s.strip().split() for s in sent]
    dic["tokens"] = [s[0] for s in sent]
    dic["gold_tags"] = [s[1] for s in sent]
    return dic


def load_data(filename):
    return [
        (data["tokens"], data["gold_tags"])
        for data in read_data("A4-data/train")
    ]


def reverse_map(_map):
    reversed_map = {}
    for key, val in _map.items():
        reversed_map[val] = key
    return reversed_map


def get_max_word_len(word_vocab):
    max_len = 0
    for _, val in word_vocab.idx2token.items():
        max_len = max(max_len, len(val))
    return max_len


print("==================Loading Data=======================")
# Make up some training data
train_data = load_data("A4-data/train")[:]
dev_data = load_data("A4-data/dev.answers")[:]
test_data = load_data("A4-data/test_answers/test.answers")[:]

train_data = sample(train_data, 1000)
# dev_data = train_data
# test_data = train_data
# dev_data = sample(dev_data, 2)
# test_data = sample(test_data, 2)

train_set = BioDataset(train_data)
dev_set = BioDataset(dev_data)
test_set = BioDataset(test_data)

word_vocab = Vocab(train_set.X, base_map={PADDING: 0, UNK_TOKEN: 1})
tag_vocab = Vocab(
    train_set.y,
    base_map={
        START_TAG: 0,
        STOP_TAG: 1,
        "O": 2,
        "B-DNA": 3,
        "I-DNA": 4,
        "B-RNA": 5,
        "I-RNA": 6,
        "B-protein": 7,
        "I-protein": 8,
        "B-cell_line": 9,
        "I-cell_line": 10,
        "B-cell_type": 11,
        "I-cell_type": 12,
    },
)
char_vocab = Vocab(
    [word for sequence in train_set.X for word in sequence],
    base_map={PADDING: 0, UNK_TOKEN: 1},
)

max_word_len = get_max_word_len(word_vocab)

# text preprocess pipeline
def text_pipeline(sentence):
    return [word_vocab.lookup_index(token) for token in sentence]


# label preprocess pipeline
def label_pipeline(tags):
    return [tag_vocab.lookup_index(tag) for tag in tags]


# preprocess batch data before loading each batch
def collate_batch(batch):
    label_list, text_list, index_list = [], [], []
    for _text, _label, _index in batch:
        text_list.append(torch.tensor(text_pipeline(_text), dtype=torch.long))
        label_list.append(
            torch.tensor(label_pipeline(_label), dtype=torch.long)
        )
        index_list.append(torch.tensor(_index, dtype=torch.long))

    len_list = torch.tensor(list(map(len, text_list)), dtype=torch.long)
    text_list = pad_sequence(text_list, batch_first=True, padding_value=0)
    label_list = pad_sequence(label_list, batch_first=True, padding_value=-1)
    index_list = torch.tensor(index_list, dtype=torch.long)

    # sort the batch according to the sequence length in the descending order
    len_list, perm_idx = len_list.sort(0, descending=True)
    text_list = text_list[perm_idx]
    label_list = label_list[perm_idx]
    index_list = index_list[perm_idx]

    return text_list.to(DEVICE), label_list.to(DEVICE), len_list, index_list


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
