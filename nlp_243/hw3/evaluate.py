import torch
from sklearn import metrics

from dataset import convert_to_token, process_output_corpus

# Evaluation
def evaluate(true, pred, output_dict=False):
    gold = [slot for sequence in true for slot in sequence]
    pred = [slot for sequence in pred for slot in sequence]
    return metrics.classification_report(
        gold, pred, output_dict=output_dict, zero_division=0
    )


def inference(model, val_X):
    val_pred = model(val_X)
    val_pred = torch.argmax(val_pred, dim=1)
    return val_pred


def joint_accuracy(val_pred, val_set, utterance_slot_dataset):
    input_seqs = [
        utterance_slot_dataset.unpad_utterances[indice]
        for indice in val_set.indices
    ]
    trues = [
        utterance_slot_dataset.unpad_slots[indice]
        for indice in val_set.indices
    ]
    preds = convert_to_token(
        val_pred.tolist(), utterance_slot_dataset.idx2slot
    )

    input_seqs, preds, trues = process_output_corpus(input_seqs, preds, trues)

    correct_count = 0
    for i in range(len(input_seqs)):
        current_count = 0
        for j in range(len(preds[i])):
            if preds[i][j] == trues[i][j]:
                current_count += 1
        if current_count == len(preds[i]):
            correct_count += 1

    return correct_count / len(input_seqs)


def evaluation_report(val_pred, val_set, utterance_slot_dataset):

    input_seqs = [
        utterance_slot_dataset.unpad_utterances[indice]
        for indice in val_set.indices
    ]
    trues = [
        utterance_slot_dataset.unpad_slots[indice]
        for indice in val_set.indices
    ]
    preds = convert_to_token(
        val_pred.tolist(), utterance_slot_dataset.idx2slot
    )

    input_seqs, preds, trues = process_output_corpus(input_seqs, preds, trues)

    return evaluate(trues, preds, output_dict=True)
