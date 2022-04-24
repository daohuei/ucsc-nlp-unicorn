import math
import time

import torch
from tqdm.auto import tqdm
from transformers import AdamW
from transformers import get_scheduler

from model import (
    DistilBERTIntentModel,
    DistilBERTSlotModel,
    ALBERTIntentModel,
    ALBERTSlotModel,
    count_parameters,
)
from config import SEED, MODEL, TASK, NAME, DEVICE, NUM_EPOCHS, LR
from helper import print_stage, epoch_time, write_predictions, write_scores
from data import get_data_loader, intent_vocab, slot_vocab, dev_true
from plot import init_report, plot_loss, plot_f1, plot_accuracy
from auto_evaluation import f1_score, accuracy_score
from inference import inference

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def experiment_pipeline(name):
    print_stage(name)

    print_stage("Loading Data")
    train_dataloader = get_data_loader("train")
    dev_dataloader = get_data_loader("dev")
    test_dataloader = get_data_loader("test")

    print_stage("Modeling")
    model = None
    if MODEL == "distil_bert" and TASK == "slot":
        model = DistilBERTSlotModel()
    if MODEL == "distil_bert" and TASK == "intent":
        model = DistilBERTIntentModel()
    if MODEL == "albert" and TASK == "slot":
        model = ALBERTSlotModel()
    if MODEL == "albert" and TASK == "intent":
        model = ALBERTIntentModel()
    model.to(DEVICE)
    print(f"The model has {count_parameters(model):,} trainable parameters")

    optimizer = AdamW(model.parameters(), lr=LR)

    # define the optimizer and learning rate scheduler
    num_training_steps = NUM_EPOCHS * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    best_dev_f1 = float("-inf")
    best_dev_accuracy = None

    train_report = init_report()
    dev_report = init_report()

    print_stage(f"Training {name}")
    for epoch in range(1, NUM_EPOCHS + 1):

        start_time = time.time()

        train_loss = train(model, train_dataloader, optimizer, lr_scheduler)
        dev_loss, dev_output = evaluate(model, dev_dataloader)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        # convert output to predictions
        dev_predictions = []
        for dev_output_val in dev_output:
            dev_pred = None
            if TASK == "intent":
                dev_pred_binary = torch.where(dev_output_val >= 0.5, 1, 0)
                dev_pred = [
                    intent_vocab.lookup_token(intent_idx)
                    for intent_idx, val in enumerate(dev_pred_binary)
                    if val.item()
                ]

            elif TASK == "slot":
                dev_pred_indexes = dev_output_val.argmax(-1)
                dev_pred = [
                    slot_vocab.lookup_token(idx.item())
                    for idx in dev_pred_indexes
                ]

            assert dev_pred != None
            dev_predictions.append(dev_pred)

        for i in range(len(dev_true)):
            assert len(dev_true[i]) == len(dev_predictions[i])

        # calculate f1-score
        dev_f1 = float("-inf")
        if TASK == "intent":
            dev_f1 = f1_score(dev_true, dev_predictions, intent=True)
        elif TASK == "slot":
            dev_f1 = f1_score(dev_true, dev_predictions)

        dev_accuracy = accuracy_score(dev_true, dev_predictions)

        if best_dev_f1 < dev_f1:
            best_dev_f1 = dev_f1
            best_dev_accuracy = dev_accuracy
            write_scores(
                {"f1": [best_dev_f1], "accuracy": [best_dev_accuracy]},
                "dev",
                name,
            )
            torch.save(model.state_dict(), f"{name}-best-model.pt")

        train_report["epoch"].append(epoch)
        train_report["loss"].append(train_loss)

        dev_report["epoch"].append(epoch)
        dev_report["loss"].append(dev_loss)
        dev_report["f1"].append(dev_f1)
        dev_report["accuracy"].append(dev_accuracy)

        plot_loss(train_report, "train", False, name)
        plot_loss(dev_report, "dev", False, name)
        plot_f1(dev_report, "dev", False, name)
        plot_accuracy(dev_report, "dev", False, name)

        print(f"Training {name}")
        print(f"Epoch: {epoch:02} | Time: {epoch_mins}m {epoch_secs}s")
        print(f"\tTrain Loss: {train_loss:.3f}")
        print(f"\tDev Loss: {dev_loss:.3f}")
        print(f"\tDev F-1: {dev_f1:.3f}")
        print(f"\tDev Accuracy: {dev_accuracy:.3f}")

    print_stage("Evaluating Test Set")
    test_predictions = inference(model, test_dataloader)
    write_predictions(test_predictions, "test", name)


def train(model, loader, optimizer, lr_scheduler):
    model.train()

    epoch_loss = 0

    # only being used in intent multi-label classification task
    bce_criterion = torch.nn.BCELoss()

    for batch in tqdm(loader):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        X = batch["input_ids"]
        y = batch["labels"]
        mask = batch["attention_mask"]

        input_batch = {
            "input_ids": X,
            "labels": y,
            "attention_mask": mask,
        }

        # forwarding
        outputs = None
        if TASK == "slot":
            outputs = model(**input_batch)
        elif TASK == "intent":
            outputs = model(X, mask)

        # calculating loss
        loss = None
        if TASK == "slot":
            # just use the loss provided by HuggingFace output
            loss = outputs.loss
        elif TASK == "intent":
            # Multi-label should use BCE instead
            loss = bce_criterion(outputs, y.type(torch.float))
        assert loss != None

        # Back Propagation
        loss.backward()

        # update the weight
        optimizer.step()

        # update the learning rate
        lr_scheduler.step()

        # clear the grad
        optimizer.zero_grad()

        epoch_loss += loss.item()

    return epoch_loss / len(loader)


def evaluate(model, loader):
    model.eval()

    epoch_loss = 0

    # only being used in intent multi-label classification task
    bce_criterion = torch.nn.BCELoss()
    output_list = []
    with torch.no_grad():
        for batch in tqdm(loader):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            X = batch["input_ids"]
            y = batch["labels"]
            mask = batch["attention_mask"]
            word_ids = batch["word_ids"]

            input_batch = {
                "input_ids": X,
                "labels": y,
                "attention_mask": mask,
            }
            # forwarding
            outputs, output_val = None, None
            if TASK == "slot":
                outputs = model(**input_batch)
                output_val = outputs.logits
                for batch_idx in range(output_val.shape[0]):
                    word_id = word_ids[batch_idx, :]
                    correct_output_idx = []
                    prev_w_id = None
                    for idx, w_id in enumerate(word_id):
                        if w_id.item() == prev_w_id:
                            continue
                        prev_w_id = w_id.item()
                        if math.isnan(w_id.item()):
                            continue
                        correct_output_idx.append(idx)
                    output_list.append(
                        output_val[batch_idx, :, :][correct_output_idx]
                    )
            elif TASK == "intent":
                outputs = model(X, mask)
                output_val = outputs
                for batch_idx in range(output_val.shape[0]):
                    output_list.append(output_val[batch_idx, :])

            # calculating loss
            loss = None
            if TASK == "slot":
                # just use the loss provided by HuggingFace output
                loss = outputs.loss
            elif TASK == "intent":
                # Multi-label should use BCE instead
                loss = bce_criterion(outputs, y.type(torch.float))
            assert loss != None

            epoch_loss += loss.item()

    return epoch_loss / len(loader), output_list


if __name__ == "__main__":
    experiment_pipeline(NAME)
