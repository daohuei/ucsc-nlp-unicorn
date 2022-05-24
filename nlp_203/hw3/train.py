import os

from transformers import default_data_collator
from transformers import (
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
)

from config import MODEL, NAME, BATCH_SIZE, DEVICE, IS_FINE_TUNE
from data import tokenizer, load_covid_data, prepare_dataset
from inference import inference, write_dict_to_json
from helper import print_stage


def train(name, fine_tune=False):
    print_stage("Loading Data")
    dev_dataset = load_covid_data(split="dev")
    test_dataset = load_covid_data(split="test")
    train_dataset = (
        load_covid_data(split="train") if fine_tune else dev_dataset
    )

    print_stage("Tokenizing")
    tokenized_dev_dataset = prepare_dataset(dev_dataset, prepare_type="train")
    tokenized_train_dataset = (
        prepare_dataset(train_dataset, prepare_type="train")
        if fine_tune
        else tokenized_dev_dataset
    )

    # get data collator
    data_collator = default_data_collator

    print_stage("Loading Model")
    # load model
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL).to(DEVICE)

    # setup trainer
    args = TrainingArguments(
        name,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=10,
        weight_decay=0.01,
        dataloader_num_workers=0,
        local_rank=-1,
        # resume_from_checkpoint=f"{name}/checkpoint-19000",
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_dev_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    if fine_tune:
        print_stage("Fine-tuning")
        trainer.train()

    print_stage("Inferencing on Dev set")
    dev_features = prepare_dataset(dev_dataset, prepare_type="eval")
    dev_pred_dict = inference(trainer, dev_features, dev_dataset)
    write_dict_to_json(dev_pred_dict, f"{name}_dev_pred.json")

    print_stage("Inferencing on Test set")
    test_features = prepare_dataset(test_dataset, prepare_type="eval")
    test_pred_dict = inference(trainer, test_features, test_dataset)
    write_dict_to_json(test_pred_dict, f"{name}_test_pred.json")


if __name__ == "__main__":
    train(NAME, fine_tune=IS_FINE_TUNE)
