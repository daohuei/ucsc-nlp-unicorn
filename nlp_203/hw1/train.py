import math
import time
import torch
import torch.nn as nn
import torch.optim as optim

from constant import DEVICE
from model import Attention, Encoder, Decoder, Seq2Seq
from data import get_data_loader, word_vocab
from helper import epoch_time, print_stage
from plot import init_report, plot_loss_ppl


def init_weights(m):
    for name, param in m.named_parameters():
        if "weight" in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def training_pipeline():
    print_stage("Modeling")
    INPUT_DIM = len(word_vocab)
    OUTPUT_DIM = len(word_vocab)
    ENC_EMB_DIM = 256  # 256
    DEC_EMB_DIM = 256  # 256
    ENC_HID_DIM = 512  # 512
    DEC_HID_DIM = 512  # 512
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5
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

    print_stage("Training")
    N_EPOCHS = 100
    CLIP = 1
    BATCH_SIZE = 4

    # Load the data loader
    train_loader = get_data_loader(BATCH_SIZE, "train")
    # val_loader = get_data_loader(BATCH_SIZE, "dev")
    # test_loader = get_data_loader(BATCH_SIZE, "test")

    best_valid_loss = float("inf")
    train_report = init_report()
    valid_report = init_report()

    for epoch in range(N_EPOCHS):

        start_time = time.time()

        train_loss = train(model, train_loader, optimizer, criterion, CLIP)
        valid_loss, _ = evaluate(model, val_loader, criterion, is_score=True)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), "tut4-model.pt")

        train_ppl = math.exp(train_loss)
        train_report["epoch"].append(epoch)
        train_report["loss"].append(train_loss)
        train_report["perplexity"].append(train_ppl)
        plot_loss_ppl(train_report, "train", True)

        valid_ppl = math.exp(valid_loss)
        valid_report["epoch"].append(epoch)
        valid_report["loss"].append(valid_loss)
        valid_report["perplexity"].append(valid_ppl)
        plot_loss_ppl(valid_report, "dev", True)

        # for k in ["1", "2", "l"]:
        #     for m in ["precision", "recall", "f1"]:
        #         valid_report[f"rouge-{k}-{m}"].append(
        #             rouge_scores[f"rouge-{k}-{m[0]}"]
        #         )
        # plot_rouge(valid_report, "dev", True)

        print(f"Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s")
        print(f"\tTrain Loss: {train_loss:.3f} | Train PPL: {train_ppl:7.3f}")
        print(f"\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {valid_ppl:7.3f}")

    # model.load_state_dict(torch.load("tut4-model.pt"))

    # test_loss = evaluate(model, test_loader, criterion)

    # print(
    #     f"| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |"
    # )


# train the model
def train(model, loader, optimizer, criterion, clip):
    model.train()

    epoch_loss = 0

    for _, batch in enumerate(loader):

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

    with torch.no_grad():

        for _, batch in enumerate(loader):

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

            # remove START token and combine batch and seq len dim
            output = output[1:].view(-1, output_dim)
            summary = summary[1:].view(-1)

            # summary = [(summary len - 1) * batch size]
            # output = [(summary len - 1) * batch size, decoder output dim]

            # calculate the loss
            loss = criterion(output, summary)

            epoch_loss += loss.item()

    return epoch_loss / len(loader)


if __name__ == "__main__":
    training_pipeline()