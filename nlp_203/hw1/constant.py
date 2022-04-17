from gpu import device

SRC_FILENAME = {
    "train": "cnndm/data/train.txt.src",
    "train_truncate": "cnndm/data/truncated_train.txt.src",
    "dev": "cnndm/data/val.txt.src",
    "test": "cnndm/data/test.txt.src",
}
TGT_FILENAME = {
    "train": "cnndm/data/train.txt.tgt",
    "dev": "cnndm/data/val.txt.tgt",
    "test": "cnndm/data/test.txt.tgt",
}

START = "<START>"
STOP = "<STOP>"
PADDING = "<PAD>"
UNK_TOKEN = "<UNK>"
DEVICE = device
TRUNCATE_SIZE = 10
