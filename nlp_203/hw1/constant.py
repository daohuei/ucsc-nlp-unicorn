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
TRUNCATE_SIZE = 512
MAX_VOCAB_SIZE = 512
LEAST_FREQ = 0
BATCH_SIZE = 8
ENC_EMB_DIM = 64  # 256
DEC_EMB_DIM = 64  # 256
ENC_HID_DIM = 128  # 512
DEC_HID_DIM = 128  # 512
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
HYPER_KEY = f"batch_{BATCH_SIZE}_enc_emb_{ENC_EMB_DIM}_hid_{ENC_HID_DIM}_dec_emb_{DEC_EMB_DIM}_hid_{DEC_HID_DIM}_truncate_{TRUNCATE_SIZE}_vocab_{MAX_VOCAB_SIZE}_freq_{LEAST_FREQ}"
NAME = f"seq2seq_{HYPER_KEY}"
