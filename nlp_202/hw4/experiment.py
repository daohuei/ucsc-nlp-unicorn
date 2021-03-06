import torch
import torch.optim as optim

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
from data import (
    get_data_loader,
    get_sampled_data_loader,
    word_vocab,
    tag_vocab,
)
from constants import DEVICE
from helper import hamming_loss

word_to_ix = word_vocab.token2idx
tag_to_ix = tag_vocab.token2idx


def experiment(
    emb_dim=5,
    char_emb_dim=4,
    char_cnn_stride=2,
    char_cnn_kernel=2,
    hidden_dim=4,
    epoch_num=2,
    batch_size=2,
    lr=0.01,
    lamb=1e-4,
    name="model",
    char_cnn=False,
    loss="crf_loss",
    resume=False,
    cost_val=10,
):

    # use data loader for batching data
    train_loader = get_data_loader(batch_size=batch_size, set_name="train")
    dev_loader = get_data_loader(batch_size=batch_size, set_name="dev")
    test_loader = get_data_loader(batch_size=batch_size, set_name="test")

    # train_loader = get_sampled_data_loader(
    #     batch_size=batch_size, set_name="train"
    # )
    # dev_loader = get_sampled_data_loader(batch_size=batch_size, set_name="dev")
    # test_loader = get_sampled_data_loader(
    #     batch_size=batch_size, set_name="test"
    # )

    print("======================Initializing Model=================")
    model = BiLSTM_CRF(
        len(word_vocab),
        tag_vocab.token2idx,
        emb_dim,
        hidden_dim,
        char_cnn=char_cnn,
        char_cnn_stride=char_cnn_stride,
        char_cnn_kernel=char_cnn_kernel,
        char_embedding_dim=char_emb_dim,
        loss=loss,
        cost=hamming_loss(loss_val=cost_val),
    ).to(DEVICE)

    prev_best_score = None
    if resume:
        model.load_state_dict(torch.load(f"{name}.pt", map_location=DEVICE))
        dev_all_input, dev_all_preds, dev_all_golds = inference(
            model, dev_loader
        )
        prev_best_score = batch_evaluate(dev_all_golds, dev_all_preds)
    # optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=lamb)
    optimizer = optim.Adam(model.parameters(), lr=0.1, weight_decay=lamb)

    print("======================Training Model=================")
    (model, train_epoch_times) = train(
        model,
        optimizer,
        train_loader,
        dev_loader,
        epoch_num,
        name=name,
        prev_best_score=prev_best_score,
    )

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
        batch_size,
        sum(train_epoch_times) / len(train_epoch_times),
        name=f"{name}.time",
    )
