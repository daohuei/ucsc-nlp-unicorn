import time

import torch
from tqdm import tqdm
from matplotlib import pyplot as plt

from helper import unpad_sequence, convert_batch_sequence
from data import tag_vocab
from evaluate import batch_evaluate

EARLY_STOPPING_THRES = 3


def train(
    model,
    optimizer,
    train_loader,
    dev_loader,
    epoch_num,
    name="model_name",
    prev_best_score=None,
):
    # record loss in every epoch for train and dev set for plotting
    avg_train_epoch_losses = []
    train_epoch_times = []

    # plotting loss over time for debugging training process
    loss_ax, loss_fig = init_loss_plot()

    # best score: precision, recall, f-1
    best_score = (float("-inf"), float("-inf"), float("-inf"))
    if prev_best_score:
        best_score = prev_best_score

    # for checking early stopping
    no_improve_count = 0

    train_size = len(train_loader.dataset)
    dev_size = len(dev_loader.dataset)
    for epoch in range(1, epoch_num + 1):
        # for recording training time
        # start of training
        start_time = time.time()

        epoch_train_loss = 0

        # training
        train_preds = []
        train_golds = []
        for X, Y, seq_lens, _ in tqdm(train_loader, desc="Training"):
            model.zero_grad()

            # Run our forward pass and compute the loss
            loss = model.neg_log_likelihood(X, Y, seq_lens)

            epoch_train_loss += loss.item()
            # Compute gradients with loss
            loss.backward()

            # Update the parameters by optimizer.step()
            optimizer.step()

        # record the average loss
        avg_train_epoch_loss = epoch_train_loss / train_size
        avg_train_epoch_losses.append(avg_train_epoch_loss)

        # end of training
        end_time = time.time()
        train_epoch_times.append(end_time - start_time)

        # Evaluating
        dev_preds = []
        dev_golds = []
        for X, Y, seq_lens, _ in tqdm(dev_loader, desc="Validating"):
            # making prediction on dev set and store the prediction
            _, preds = model.forward(X, seq_lens)
            golds = unpad_sequence(Y.cpu().numpy(), seq_lens)
            dev_preds += preds
            dev_golds += golds

        # evaluate the dev score
        dev_preds = convert_batch_sequence(dev_preds, tag_vocab)
        dev_golds = convert_batch_sequence(dev_golds, tag_vocab)
        dev_precision, dev_recall, dev_f1 = batch_evaluate(
            dev_golds, dev_preds
        )

        # print the performance of current epoch
        print(f"Epoch {epoch} Training Loss: {avg_train_epoch_loss}")
        print(f"Epoch {epoch}  Dev F-1: {dev_f1}")
        plot_losses(loss_ax, epoch, avg_train_epoch_losses)
        loss_fig.savefig(f"{name}_loss.png")

        # store the best model by evaluating the score
        best_f1 = best_score[2]
        if best_f1 < dev_f1:
            no_improve_count = 0
            best_score = (dev_precision, dev_recall, dev_f1)
            torch.save(model.state_dict(), f"{name}.pt")
        else:
            # if not improving, early stop the training process
            no_improve_count += 1
            if no_improve_count >= EARLY_STOPPING_THRES and best_f1 > 0:
                print("Not improving, early stopped!!")
                break

    model.load_state_dict(torch.load(f"{name}.pt"))
    return (
        model,
        train_epoch_times,
    )


def init_loss_plot():
    loss_fig, loss_ax = plt.subplots(1, 1, figsize=(15, 5))

    # Plot the comparison between training time and batch size
    loss_ax.set_xlabel("Epochs")
    loss_ax.set_ylabel("Loss")
    loss_ax.set_title("Train Loss over Epochs")
    loss_ax.plot([], [], "r", label="train loss")

    loss_ax.legend()
    return loss_ax, loss_fig


def plot_losses(loss_ax, epoch_num, avg_train_epoch_losses):
    loss_ax.plot(
        list(range(1, epoch_num + 1)),
        avg_train_epoch_losses,
        "r",
        label="train loss",
    )
