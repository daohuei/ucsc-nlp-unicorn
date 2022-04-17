import random

import typer
import torch
import numpy as np

from data import read_data


SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

app = typer.Typer()


@app.command()
def main():
    read_data()


if __name__ == "__main__":
    app()
