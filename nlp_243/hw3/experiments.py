import sys

import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import random_split, DataLoader

from utils import (
    read_glove_vector,
    get_one_hot_matrix,
    get_glove_matrix,
    create_emb_layer,
)
from dataset import UtteranceSlotDataset
from train import train
from gpu import device
from models import BaselineModel, RNNTwoLayerModel, GRUModel, LSTMModel

# Get the raw data as pandas DataFrame
train_df = pd.read_csv("hw_3_train_data.csv")


# Use dataset object for preprocessing the raw data
train_utterances = list(train_df["utterances"])
train_slots = list(train_df["IOB Slot tags"])
utterance_slot_dataset = UtteranceSlotDataset(
    train_utterances,
    train_slots,
    seq_len=int(np.max([len(sent.split()) for sent in train_utterances])) + 10,
)

# split the training data into training set and validation set
val_len = int(len(utterance_slot_dataset) * 0.3)
train_set, val_set = random_split(
    utterance_slot_dataset, [len(utterance_slot_dataset) - val_len, val_len]
)

# Define Global hyperparameters

## Model
num_classes = len(utterance_slot_dataset.slot2idx)
seq_len = utterance_slot_dataset.seq_len

## Training
batch_size = 2048

# with splitting (for validation)
train_loader = DataLoader(
    dataset=train_set, batch_size=batch_size, shuffle=True,
)
# without splitting (for output test result)
all_train_loader = DataLoader(
    dataset=utterance_slot_dataset, batch_size=batch_size, shuffle=True,
)


if __name__ == "__main__":
    args = sys.argv
    embedding = args[1]
    model_type = args[2]
    train_mode = args[3]
    n_epochs = int(args[4])

    weight_matrix = None
    if embedding == "one_hot":
        weight_matrix = get_one_hot_matrix(utterance_slot_dataset.vocab)
    elif embedding == "glove":
        glove_map = read_glove_vector("glove.6B.50d.txt")
        weight_matrix = get_glove_matrix(
            glove_map, utterance_slot_dataset.vocab
        )
    elif embedding == "glove_100d":
        glove_100d_map = read_glove_vector("glove.6B.100d.txt")
        weight_matrix = get_glove_matrix(
            glove_100d_map, utterance_slot_dataset.vocab
        )

    # create the embedding layer
    emb_layer, num_embeddings, embedding_dim = create_emb_layer(weight_matrix)

    # select models
    model = None
    if model_type == "baseline_rnn":
        model = BaselineModel(
            input_size=embedding_dim,
            output_size=num_classes,
            seq_len=seq_len,
            emb_layer=emb_layer,
        ).to(device)
    elif model_type == "2_layer_rnn":
        model = RNNTwoLayerModel(
            input_size=embedding_dim,
            hidden_size=32,
            output_size=num_classes,
            seq_len=seq_len,
            emb_layer=emb_layer,
        ).to(device)
    elif model_type == "gru":
        model = GRUModel(
            input_size=embedding_dim,
            output_size=num_classes,
            seq_len=seq_len,
            emb_layer=emb_layer,
        ).to(device)
    elif model_type == "lstm":
        model = LSTMModel(
            input_size=embedding_dim,
            output_size=num_classes,
            seq_len=seq_len,
            emb_layer=emb_layer,
        ).to(device)
    loader = None
    if train_mode == "validate":
        loader = train_loader
    elif train_mode == "all":
        loader = all_train_loader

    reports = train(
        model=model,
        n_epochs=n_epochs,
        data_loader=loader,
        loss_func=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=0.005),
        val_set=val_set,
        dataset=utterance_slot_dataset,
        is_plot=True,
        plot_name=f"{model_type}_{embedding}",
    )
    best_idx = int(np.argmax([report[1]["accuracy"] for report in reports]))
    final_val_report = reports[best_idx][1]
    final_val_joint_accuracy = reports[best_idx][3]
    print("Accuracy: ", final_val_report["accuracy"])
    print("Macro F1-Score: ", final_val_report["macro avg"]["f1-score"])
    print("Weighted F1-Score: ", final_val_report["weighted avg"]["f1-score"])
    print("Joint Accuracy: ", final_val_joint_accuracy)
    print("Best Epoch: ", best_idx * 10)
