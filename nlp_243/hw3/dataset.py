import torch
from torch.utils.data import Dataset

from tokens import *
from utils import *
from gpu import device


# Data Preprocessing
# Create Dataset class
class UtteranceSlotDataset(Dataset):
    def __init__(self, utterances, slots, seq_len=None):
        # tokenization
        utterances = tokenize(utterances)
        slots = tokenize(slots)

        # adding start and stop tokens
        utterances = add_start_stop_tokens(utterances)
        slots = add_start_stop_tokens(slots)

        # padding
        self.seq_len = seq_len
        if not seq_len:
            self.seq_len = len(max(utterances, key=lambda sent: len(sent)))
        utterances = padding(utterances, self.seq_len)
        slots = padding(slots, self.seq_len)

        # get vocab for both utterance and slot
        self.vocab, self.word2idx, self.idx2word = build_vocab(utterances)
        self.slot_list, self.slot2idx, self.idx2slot = build_vocab(slots)

        # convert to index space
        utterances = convert_to_idx(utterances, self.word2idx)
        slots = convert_to_idx(slots, self.slot2idx)

        # Convert arrays to torch tensors
        self.X = torch.tensor(utterances).to(device)
        self.y = torch.tensor(slots).to(device)

    # Must have
    def __len__(self):
        return len(self.y)

    # Must have
    def __getitem__(self, index):
        return self.X[index], self.y[index]
