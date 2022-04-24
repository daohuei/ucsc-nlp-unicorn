import matplotlib.pyplot as plt

METRICS = ["epoch", "loss"]
SPLIT_COLOR_MAP = {
    "train": "b",
    "dev": "r",
}

LINE_STYLE_MAP = {
    "precision": "--",
    "recall": ":",
    "f1": "-",
}


def init_report():
    return {metric: [] for metric in METRICS}


def init_fig(title, x_label, y_label, legend=True):

    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    # ax.set_ylim([0, 100000])
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if legend:
        ax.plot([], [], SPLIT_COLOR_MAP["train"], label=f"train {y_label}")
        ax.plot([], [], SPLIT_COLOR_MAP["dev"], label=f"dev {y_label}")
        ax.legend()
    return fig, ax


fig_loss, ax_loss = init_fig("Loss per Epoch", "Epoch", "Loss")


def plot_loss(report, split, periodically=False, name=""):
    epochs = report["epoch"]
    if periodically and len(epochs) % 10 != 0:
        return
    color = SPLIT_COLOR_MAP[split]
    losses = report["loss"]
    ax_loss.plot(epochs, losses, f"{color}", label=f"{split}-loss")
    fig_loss.savefig(f"{name}_loss_fig.jpg")
