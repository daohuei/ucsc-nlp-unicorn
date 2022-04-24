import matplotlib.pyplot as plt

METRICS = ["epoch", "loss", "f1", "accuracy"]
SPLIT_COLOR_MAP = {
    "train": "b",
    "dev": "r",
}


def init_report():
    return {metric: [] for metric in METRICS}


def init_fig(title, x_label, y_label, legend=True):

    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if legend:
        ax.plot([], [], SPLIT_COLOR_MAP["train"], label=f"train {y_label}")
        ax.plot([], [], SPLIT_COLOR_MAP["dev"], label=f"dev {y_label}")
        ax.legend()
    return fig, ax


fig_loss, ax_loss = init_fig("Loss per Epoch", "Epoch", "Loss")
fig_f1, ax_f1 = init_fig("F-1 per Epoch", "Epoch", "F1")
fig_accuracy, ax_accuracy = init_fig("Accuracy per Epoch", "Epoch", "Accuracy")


def plot_loss(report, split, periodically=False, name=""):
    epochs = report["epoch"]
    if periodically and len(epochs) % 10 != 0:
        return
    color = SPLIT_COLOR_MAP[split]
    losses = report["loss"]
    ax_loss.plot(epochs, losses, f"{color}", label=f"{split}-loss")
    fig_loss.savefig(f"{name}_loss_fig.jpg")


def plot_f1(report, split, periodically=False, name=""):
    epochs = report["epoch"]
    if periodically and len(epochs) % 10 != 0:
        return
    color = SPLIT_COLOR_MAP[split]
    f1s = report["f1"]
    ax_f1.plot(epochs, f1s, f"{color}", label=f"{split}-f1")
    fig_f1.savefig(f"{name}_f1_fig.jpg")


def plot_accuracy(report, split, periodically=False, name=""):
    epochs = report["epoch"]
    if periodically and len(epochs) % 10 != 0:
        return
    color = SPLIT_COLOR_MAP[split]
    accuracy_list = report["accuracy"]
    ax_accuracy.plot(
        epochs, accuracy_list, f"{color}", label=f"{split}-accuracy"
    )
    fig_accuracy.savefig(f"{name}_accuracy_fig.jpg")
