import torch
from tqdm import tqdm
import pandas as pd

from conlleval import evaluate as conllevaluate
from helper import convert_batch_sequence, unpad_sequence
from data import word_vocab, tag_vocab


def batch_evaluate(golds, preds, verbose=True):
    all_golds = [tag for gold in golds for tag in gold]
    all_preds = [tag for pred in preds for tag in pred]

    return evaluate(all_golds, all_preds)


def evaluate(all_gold_tags, all_predicted_tags, verbose=True):
    return conllevaluate(all_gold_tags, all_predicted_tags, verbose)


def inference(model, data_loader):
    all_input = []
    all_preds = []
    all_golds = []
    raw_inputs = data_loader.dataset.X
    for X, Y, seq_lens, indexes in tqdm(data_loader, desc="Inference"):

        # making prediction on dev set and store the prediction
        _, preds = model.forward(X, seq_lens)
        golds = unpad_sequence(Y.numpy(), seq_lens)
        inputs = [raw_inputs[index] for index in indexes]

        all_input += inputs
        all_preds += preds
        all_golds += golds

    all_preds = convert_batch_sequence(all_preds, tag_vocab)
    all_golds = convert_batch_sequence(all_golds, tag_vocab)
    return all_input, all_preds, all_golds


def output_prediction(all_input, all_preds, all_golds, name="model"):
    assert len(all_input) == len(all_preds)
    assert len(all_input) == len(all_golds)

    with open(name, "w", encoding="utf-8") as f:
        for i in range(len(all_input)):
            input_seq = all_input[i]
            preds = all_preds[i]
            golds = all_golds[i]

            assert len(input_seq) == len(preds)
            assert len(input_seq) == len(golds)

            # output the submission
            for j in range(len(input_seq)):
                token = input_seq[j]
                line = f"{token}\t{golds[j]}\t{preds[j]}\n"
                f.write(line)
            f.write("\n")


def output_report(precision, recall, f1, name="model.dev.report"):
    report_df = pd.DataFrame(
        data={"precision": [precision], "recall": [recall], "f1": [f1]}
    )
    report_df.to_csv(name, index=False)


def output_training_time(batch, avg_train_time, name="model.time"):
    output_df = pd.DataFrame(
        data={"batch": [batch], "training_time": [avg_train_time]}
    )
    output_df.to_csv(name, index=False)


def output_hyper_parameters(hp_map, name="model.hp"):
    output_df = pd.DataFrame(data=hp_map)
    output_df.to_csv(name, index=False)