import torch
import typer

from experiment import experiment

torch.manual_seed(1)

app = typer.Typer()


@app.command()
def bi_lstm_crf(
    name,
    emb_dim: int = 5,
    hidden_dim: int = 4,
    epoch_num: int = 2,
    batch_size: int = 2,
    lr: float = 0.01,
    lamb: float = 1e-4,
    resume: bool = False,
):
    experiment(
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        epoch_num=epoch_num,
        batch_size=batch_size,
        lr=lr,
        lamb=lamb,
        name=name,
        resume=resume,
    )


@app.command()
def bi_lstm_crf_char_cnn(
    name,
    emb_dim: int = 5,
    char_emb_dim: int = 4,
    stride: int = 2,
    kernel: int = 2,
    hidden_dim: int = 4,
    epoch_num: int = 2,
    batch_size: int = 2,
    lr: float = 0.01,
    lamb: float = 1e-4,
    resume: bool = False,
):
    experiment(
        emb_dim=emb_dim,
        char_emb_dim=char_emb_dim,
        char_cnn_stride=stride,
        char_cnn_kernel=kernel,
        hidden_dim=hidden_dim,
        epoch_num=epoch_num,
        batch_size=batch_size,
        lr=lr,
        lamb=lamb,
        name=name,
        char_cnn=True,
        resume=resume,
    )


@app.command()
def bi_lstm_crf_softmax_margin_loss(
    name,
    emb_dim: int = 5,
    hidden_dim: int = 4,
    epoch_num: int = 2,
    batch_size: int = 2,
    lr: float = 0.01,
    lamb: float = 1e-4,
    resume: bool = False,
):
    experiment(
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        epoch_num=epoch_num,
        batch_size=batch_size,
        lr=lr,
        lamb=lamb,
        name=name,
        loss="softmax_margin_loss",
        resume=resume,
    )


@app.command()
def bi_lstm_crf_svm_loss(
    name,
    emb_dim: int = 5,
    hidden_dim: int = 4,
    epoch_num: int = 2,
    batch_size: int = 2,
    lr: float = 0.01,
    lamb: float = 1e-4,
    resume: bool = False,
):
    experiment(
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        epoch_num=epoch_num,
        batch_size=batch_size,
        lr=lr,
        lamb=lamb,
        name=name,
        loss="svm_loss",
        resume=resume,
    )


@app.command()
def bi_lstm_crf_ramp_loss(
    name,
    emb_dim: int = 5,
    hidden_dim: int = 4,
    epoch_num: int = 2,
    batch_size: int = 2,
    lr: float = 0.01,
    lamb: float = 1e-4,
    resume: bool = False,
):
    experiment(
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        epoch_num=epoch_num,
        batch_size=batch_size,
        lr=lr,
        lamb=lamb,
        name=name,
        loss="ramp_loss",
        resume=resume,
    )


@app.command()
def bi_lstm_crf_soft_ramp_loss(
    name,
    emb_dim: int = 5,
    hidden_dim: int = 4,
    epoch_num: int = 2,
    batch_size: int = 2,
    lr: float = 0.01,
    lamb: float = 1e-4,
    resume: bool = False,
):
    experiment(
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        epoch_num=epoch_num,
        batch_size=batch_size,
        lr=lr,
        lamb=lamb,
        name=name,
        loss="soft_ramp_loss",
        resume=resume,
    )


if __name__ == "__main__":
    app()
