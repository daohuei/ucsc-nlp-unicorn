from gpu import device

# train: contains gold labels and input text
# test: only contains input text
# hw1_labels_dev.txt is actually just the label for train data, which is useless(not sure why it is provided)
DATA_PATH = {
    "train": "data/hw1_train.xlsx",
    "test": "data/hw1_test.xlsx",
}

SEED = 6518
UNK_TOKEN = "<UNK>"
TASK = "slot"
MODEL = "distil_bert"

BATCH_SIZE = 8
