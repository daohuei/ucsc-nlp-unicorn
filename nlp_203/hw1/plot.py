import matplotlib.pyplot as plt

METRICS = ["epoch", "loss", "perplexity"] + [
    f"rouge-{x}-{m}"
    for x in ["1", "2", "l"]
    for m in ["precision", "recall", "f1"]
]
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


def init_fig(title, x_label, y_label):

    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    # ax.set_ylim([0, 100000])
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.plot([], [], SPLIT_COLOR_MAP["train"], label=f"train {y_label}")
    ax.plot([], [], SPLIT_COLOR_MAP["dev"], label=f"dev {y_label}")
    ax.legend()
    return fig, ax


fig_loss, ax_loss = init_fig("Loss per Epoch", "Epoch", "Loss")
fig_ppl, ax_ppl = init_fig("Perplexity per Epoch", "Epoch", "PPL")

rouge_plot = {}
for k in ["1", "2", "l"]:
    rouge_plot[k] = init_fig(f"Rouge-{k} per Epoch", "Epoch", f"Rouge-{k}")


def plot_loss_ppl(report, split, periodically=False):
    epochs = report["epoch"]
    if periodically and len(epochs) % 10 != 0:
        return
    color = SPLIT_COLOR_MAP[split]
    losses = report["loss"]
    ppls = report["perplexity"]
    ax_loss.plot(epochs, losses, f"{color}", label=f"{split}-loss")
    ax_ppl.plot(epochs, ppls, f"{color}", label=f"{split}-ppl")
    fig_loss.savefig("loss_fig.jpg")
    fig_ppl.savefig("ppl_fig.jpg")


def plot_rouge(report, split, periodically=False):
    epochs = report["epoch"]
    if periodically and len(epochs) % 10 != 0:
        return
    color = SPLIT_COLOR_MAP[split]

    for k in ["1", "2", "l"]:
        for m in ["precision", "recall", "f1"]:
            key = f"rouge-{k}-{m}"
            y_values = report[key]

            rouge_fig, rouge_ax = rouge_plot[k]
            rouge_ax.plot(
                epochs,
                y_values,
                f"{color}{LINE_STYLE_MAP[m]}",
                label=f"{split}-{key}",
            )
            rouge_fig.savefig(f"{key}_fig.jpg")
