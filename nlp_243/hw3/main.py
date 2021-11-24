import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from dataset import UtteranceSlotDataset
from models import BaselineModel, create_emb_layer, get_one_hot_matrix
from train import train
from gpu import device

if __name__ == "__main__":
    train_df = pd.read_csv("hw_3_train_data.csv")
    train_utterances = list(train_df["utterances"])
    train_slots = list(train_df["IOB Slot tags"])
    utterance_slot_dataset = UtteranceSlotDataset(
        train_utterances, train_slots
    )
    val_len = int(len(utterance_slot_dataset) * 0.3)
    train_set, val_set = random_split(
        utterance_slot_dataset,
        [len(utterance_slot_dataset) - val_len, val_len],
    )

    # Define hyperparameters

    ## Model
    num_classes = len(utterance_slot_dataset.slot2idx)
    seq_len = utterance_slot_dataset.seq_len

    ## Training
    batch_size = 2048

    train_loader = DataLoader(
        dataset=train_set, batch_size=batch_size, shuffle=True,
    )

    all_train_loader = DataLoader(
        dataset=utterance_slot_dataset, batch_size=batch_size, shuffle=True,
    )
    # create the embedding layer
    one_hot_emb_layer, num_embeddings, embedding_dim = create_emb_layer(
        get_one_hot_matrix(utterance_slot_dataset.vocab)
    )

    val_model = BaselineModel(
        input_size=embedding_dim,
        output_size=num_classes,
        seq_len=seq_len,
        emb_layer=one_hot_emb_layer,
    ).to(device)
    train(
        model=val_model,
        n_epochs=2000,
        data_loader=train_loader,
        loss_func=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(val_model.parameters(), lr=0.00005),
        is_plot=True,
    )
    # create the embedding layer
    one_hot_emb_layer, num_embeddings, embedding_dim = create_emb_layer(
        get_one_hot_matrix(utterance_slot_dataset.vocab)
    )

    test_model = BaselineModel(
        input_size=embedding_dim,
        output_size=num_classes,
        seq_len=seq_len,
        emb_layer=one_hot_emb_layer,
    ).to(device)
    train(
        model=test_model,
        n_epochs=500,
        data_loader=all_train_loader,
        loss_func=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(test_model.parameters(), lr=0.00005),
        is_plot=True,
    )
    test_df = pd.read_csv("test_data.csv")
    test_utterances = list(test_df["utterances"])
    test_X = torch.tensor(
        preprocess_utterances(test_utterances, utterance_slot_dataset)
    ).to(device)
