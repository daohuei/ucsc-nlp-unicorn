import torch
import torch.optim as optim
import torch.autograd as autograd

from model import BiLSTM_CRF
from evaluate import (
    inference,
    output_prediction,
    output_report,
    output_hyper_parameters,
    output_training_time,
    batch_evaluate,
)
from train import train
from data import get_data_loader, word_vocab, tag_vocab

word_to_ix = word_vocab.token2idx
tag_to_ix = tag_vocab.token2idx


def experiment(
    emb_dim=5,
    hidden_dim=4,
    epoch_num=2,
    batch_size=2,
    lr=0.01,
    lamb=1e-4,
    name="model",
):

    # use data loader for batching data
    train_loader = get_data_loader(batch_size=batch_size, set_name="train")
    dev_loader = get_data_loader(batch_size=batch_size, set_name="dev")
    test_loader = get_data_loader(batch_size=batch_size, set_name="test")

    print("======================Declaring Model=================")
    model = BiLSTM_CRF(
        len(word_vocab), tag_vocab.token2idx, emb_dim, hidden_dim
    )
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=lamb)

    print("======================Training Model=================")
    (
        model,
        best_score,
        train_epoch_times,
    ) = train(model, optimizer, train_loader, dev_loader, epoch_num, name=name)

    print("======================Evaluating Model=================")
    dev_all_input, dev_all_preds, dev_all_golds = inference(model, dev_loader)
    dev_precision, dev_recall, dev_f1 = batch_evaluate(
        dev_all_golds, dev_all_preds
    )

    test_all_input, test_all_preds, test_all_golds = inference(
        model, test_loader
    )
    test_precision, test_recall, test_f1 = batch_evaluate(
        test_all_golds, test_all_preds
    )

    print(
        "======================Output Prediction and Report================="
    )
    output_prediction(
        dev_all_input, dev_all_preds, dev_all_golds, name=f"{name}.dev.pred"
    )
    output_report(dev_precision, dev_recall, dev_f1, name=f"{name}.dev.report")
    output_prediction(
        test_all_input,
        test_all_preds,
        test_all_golds,
        name=f"{name}.test.pred",
    )
    output_report(
        test_precision, test_recall, test_f1, name=f"{name}.test.report"
    )

    print("======================Output Hyperparameters=================")
    hp_map = {
        "emb_dim": [emb_dim],
        "hidden_dim": [hidden_dim],
        "epoch_num": [epoch_num],
        "batch_size": [batch_size],
        "lr": [lr],
        "lamb": [lamb],
    }
    output_hyper_parameters(hp_map, name=f"{name}.hp")
    output_training_time(
        batch_size, sum(train_epoch_times) / len(train_epoch_times)
    )
