import torch
import typer

from experiment import experiment

torch.manual_seed(1)

app = typer.Typer()


@app.command()
def free(
    emb_dim: int = 5,
    hidden_dim: int = 4,
    epoch_num: int = 2,
    batch_size: int = 2,
    lr: float = 0.01,
    lamb: float = 1e-4,
    name: str = "model",
):
    experiment(
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        epoch_num=epoch_num,
        batch_size=batch_size,
        lr=lr,
        lamb=lamb,
        name=name,
    )


@app.command()
def bi_lstm_crf(
    emb_dim: int = 5,
    hidden_dim: int = 4,
    epoch_num: int = 2,
    batch_size: int = 2,
    lr: float = 0.01,
    lamb: float = 1e-4,
    name: str = "model",
):
    experiment(
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        epoch_num=epoch_num,
        batch_size=batch_size,
        lr=lr,
        lamb=lamb,
        name=name,
    )


if __name__ == "__main__":
    app()