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
from config import SEED, MODEL, TASK, NAME, DEVICE, NUM_EPOCHS
from helper import print_stage, epoch_time
from data import get_data_loader
from plot import init_report, plot_loss

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def training_pipeline(name):
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

    optimizer = AdamW(model.parameters(), lr=5e-5)

    # define the optimizer and learning rate scheduler
    num_training_steps = NUM_EPOCHS * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    train_report = init_report()
    dev_report = init_report()

    print_stage(f"Training {name}")
    for epoch in range(1, NUM_EPOCHS + 1):

        start_time = time.time()

        train_loss = train(model, train_dataloader, optimizer, lr_scheduler)
        dev_loss = evaluate(model, dev_dataloader)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        train_report["epoch"].append(epoch)
        train_report["loss"].append(train_loss)

        dev_report["epoch"].append(epoch)
        dev_report["loss"].append(dev_loss)

        plot_loss(train_report, "train", False, name)
        plot_loss(dev_report, "dev", False, name)

        print(f"Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s")
        print(f"\tTrain Loss: {train_loss:.3f}")
        print(f"\tDev Loss: {dev_loss:.3f}")


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

        # forwarding
        outputs = None
        if TASK == "slot":
            outputs = model(**batch)
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

    with torch.no_grad():
        for batch in tqdm(loader):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            X = batch["input_ids"]
            y = batch["labels"]
            mask = batch["attention_mask"]

            # forwarding
            outputs = None
            if TASK == "slot":
                outputs = model(**batch)
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

            epoch_loss += loss.item()

    return epoch_loss / len(loader)


if __name__ == "__main__":
    training_pipeline(NAME)