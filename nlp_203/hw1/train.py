import math
import time

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from constant import (
    DEVICE,
    BATCH_SIZE,
    NAME,
    ENC_EMB_DIM,
    DEC_EMB_DIM,
    ENC_HID_DIM,
    DEC_HID_DIM,
    ENC_DROPOUT,
    DEC_DROPOUT,
)
from model import Attention, Encoder, Decoder, Seq2Seq
from data import get_data_loader, word_vocab, dev_data, test_data
from helper import epoch_time, print_stage, write_predictions, write_scores
from evaluate import calculate_rouges
from plot import init_report, plot_loss_ppl, plot_rouge
from inference import inference

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def init_weights(m):
    for name, param in m.named_parameters():
        if "weight" in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def training_pipeline(name="seq2seq"):
    print_stage("Modeling")
    INPUT_DIM = len(word_vocab)
    OUTPUT_DIM = len(word_vocab)
    SRC_PAD_IDX = 0
    TRG_PAD_IDX = 0

    attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
    enc = Encoder(
        INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT
    )
    dec = Decoder(
        OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn
    )

    model = Seq2Seq(enc, dec, SRC_PAD_IDX, DEVICE).to(DEVICE)

    model.apply(init_weights)

    print(model)
    print(f"The model has {count_parameters(model):,} trainable parameters")

    # define the optimizer and loss criterion
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

    print_stage(f"Training {name}")
    N_EPOCHS = 10
    CLIP = 1

    # Load the data loader
    train_loader = get_data_loader(BATCH_SIZE, "train")
    val_loader = get_data_loader(BATCH_SIZE, "dev")
    test_loader = get_data_loader(1, "test")

    dev_golds = dev_data.apply(lambda data: " ".join(data[1][1:])).tolist()
    test_golds = test_data.apply(lambda data: " ".join(data[1][1:])).tolist()

    best_valid_loss = float("inf")
    best_dev_predictions = []
    best_dev_scores = None
    train_report = init_report()
    valid_report = init_report()

    for epoch in range(N_EPOCHS):

        start_time = time.time()

        train_loss = train(model, train_loader, optimizer, criterion, CLIP)
        valid_loss, dev_output = evaluate(model, val_loader, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        dev_predictions = []
        for dev_pred in dev_output:
            dev_pred_indexes = dev_pred.argmax(1)
            dev_pred = [
                word_vocab.lookup_token(idx.item()) for idx in dev_pred_indexes
            ]
            dev_pred_str = " ".join(dev_pred[1:])
            dev_predictions.append(dev_pred_str)
        rouge_scores = calculate_rouges(dev_predictions, dev_golds)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_dev_predictions = dev_predictions
            best_dev_scores = rouge_scores
            write_predictions(best_dev_predictions, "dev", name)
            write_scores(best_dev_scores, "dev", name)
            torch.save(model.state_dict(), f"{name}-best-model.pt")

        train_ppl = math.exp(train_loss)
        train_report["epoch"].append(epoch)
        train_report["loss"].append(train_loss)
        train_report["perplexity"].append(train_ppl)
        plot_loss_ppl(train_report, "train", False, name)

        valid_ppl = math.exp(valid_loss)
        valid_report["epoch"].append(epoch)
        valid_report["loss"].append(valid_loss)
        valid_report["perplexity"].append(valid_ppl)
        plot_loss_ppl(valid_report, "dev", False, name)

        for k in ["1", "2", "l"]:
            for m in ["precision", "recall", "f1"]:
                valid_report[f"rouge-{k}-{m}"].append(
                    rouge_scores[f"rouge-{k}-{m[0]}"]
                )

        print(f"Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s")
        print(f"\tTrain Loss: {train_loss:.3f} | Train PPL: {train_ppl:7.3f}")
        print(f"\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {valid_ppl:7.3f}")

        plot_rouge(valid_report, "dev", False, name)

    print_stage("Evaluate Test Set")
    model.load_state_dict(torch.load(f"{name}-best-model.pt"))
    test_predictions = inference(model, test_loader, DEVICE)
    test_scores = calculate_rouges(test_predictions, test_golds)
    write_predictions(test_predictions, "test", name)
    write_scores(test_scores, "test", name)


# train the model
def train(model, loader, optimizer, criterion, clip):
    model.train()

    epoch_loss = 0

    for batch in tqdm(loader):

        text, summary, text_len, _ = batch

        # loaded data shape:
        # text = [batch size, text len]
        # summary = [batch size, summary len]

        batch_size = text.shape[0]

        text = text.view(-1, batch_size)
        summary = summary.view(-1, batch_size)

        # text = [text len, batch size]
        # summary = [summary len, batch size]

        optimizer.zero_grad()

        output = model(text, text_len, summary)
        output_dim = output.shape[-1]

        # output = [summary len, batch size, decoder output dim]

        # remove START token and combine batch and seq len dim
        output = output[1:].view(-1, output_dim)
        summary = summary[1:].view(-1)

        # summary = [(summary len - 1) * batch size]
        # output = [(summary len - 1) * batch size, decoder output dim]

        # calculate the loss
        loss = criterion(output, summary)

        # back prop
        loss.backward()

        # clip gradient for avoiding gradient explosion
        nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(loader)


def tensor_to_sentences(batch_token_tensor):
    return [
        " ".join(
            [
                word_vocab.lookup_token(token.item())
                for token in batch_token_tensor[:, batch_idx]
            ]
        )
        for batch_idx in range(batch_token_tensor.shape[1])
    ]


def evaluate(model, loader, criterion):

    model.eval()

    epoch_loss = 0
    outputs = []
    with torch.no_grad():

        for batch in tqdm(loader):
            text, summary, text_len, _ = batch

            # loaded data shape:
            # text = [batch size, text len]
            # summary = [batch size, summary len]

            batch_size = text.shape[0]

            text = text.view(-1, batch_size)
            summary = summary.view(-1, batch_size)

            # text = [text len, batch size]
            # summary = [summary len, batch size]

            output = model(
                text, text_len, summary, 0
            )  # turn off teacher forcing
            output_dim = output.shape[-1]
            # output = [summary len, batch size, decoder output dim]
            # pred_tokens = output.argmax(-1)
            for batch_idx in range(batch_size):
                outputs.append(output[:, batch_idx, :])

            # remove START token and combine batch and seq len dim
            output = output[1:].view(-1, output_dim)
            summary = summary[1:].view(-1)

            # summary = [(summary len - 1) * batch size]
            # output = [(summary len - 1) * batch size, decoder output dim]

            # calculate the loss
            loss = criterion(output, summary)

            epoch_loss += loss.item()

    return epoch_loss / len(loader), outputs


if __name__ == "__main__":
    print_stage(NAME)
    training_pipeline(NAME)
