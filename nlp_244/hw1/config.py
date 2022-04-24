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
DEVICE = device


# task and model setup
TASK = "slot"  # slot or intent
MODEL = "distil_bert"  # distil_bert or albert

# Hyperparameters
BATCH_SIZE = 2
NUM_EPOCHS = 3

NAME = f"{TASK}_{MODEL}_batch_{BATCH_SIZE}_epoch_{NUM_EPOCHS}"