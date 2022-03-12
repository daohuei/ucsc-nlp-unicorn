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
    pass
    # sample = word_vocab.idx2token[72]
    # print(sample)
    # processed_sample = [char_vocab.token2idx[c] for c in sample]
    # print(processed_sample)
    # print("Max Length of words: ", max_word_len)
    # processed_sample = padding_char(processed_sample, max_word_len)
    # print(processed_sample)
    # # batch * channel * sequence length
    # processed_sample = torch.tensor(
    #     processed_sample, dtype=torch.long
    # ).unsqueeze(0)
    # # print(processed_sample.unsqueeze(0).size())
    # model = CharCNN(
    #     max_word_len=max_word_len,
    # )
    # print(model(processed_sample))
    # m = nn.Conv1d(16, 33, 3, stride=2)
    # input = torch.randn(20, 16, 50)

    # print(input.size())
    # output = m(input)
    # print(output.size())


@app.command()
def bi_lstm_crf_char_cnn(
    emb_dim: int = 5,
    char_emb_dim: int = 4,
    hidden_dim: int = 4,
    epoch_num: int = 2,
    batch_size: int = 2,
    lr: float = 0.01,
    lamb: float = 1e-4,
    name: str = "bi_lstm_crf_char_cnn",
):
    experiment(
        emb_dim=emb_dim,
        char_emb_dim=char_emb_dim,
        hidden_dim=hidden_dim,
        epoch_num=epoch_num,
        batch_size=batch_size,
        lr=lr,
        lamb=lamb,
        name=name,
        char_cnn=True,
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
