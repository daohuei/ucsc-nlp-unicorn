import torch
from matplotlib import pyplot as plt

from evaluate import evaluation_report, joint_accuracy


def train(
    model,
    n_epochs,
    data_loader,
    loss_func,
    optimizer,
    val_set,
    dataset,
    is_plot=False,
    plot_name="",
):
    reports = []
    epochs_plot = []
    losses_plot = []
    train_losses_plot = []
    val_losses_plot = []
    train_accuracy_plot = []
    val_accuracy_plot = []
    train_joint_plot = []
    val_joint_plot = []

    for epoch in range(1, n_epochs + 1):

        for batch_idx, (x, true_y) in enumerate(data_loader):
            # reset grads in the optimizer
            optimizer.zero_grad()
            # make prediction
            pred_y = model(x)
            # calculate the loss
            loss = loss_func(pred_y, true_y)
            # back propagation
            loss.backward()
            # perform optimization step
            optimizer.step()
        # perform evaluation on val set every 10 epochs
        if epoch % 10 == 0:
            print(f"{epoch}/{n_epochs}:", end="\t")

            train_X = dataset.X[data_loader.dataset.indices]
            train_y = dataset.y[data_loader.dataset.indices]

            val_X = dataset.X[val_set.indices]
            val_y = dataset.y[val_set.indices]

            train_pred = model(train_X)
            val_pred = model(val_X)

            train_loss = loss_func(train_pred, train_y)
            val_loss = loss_func(val_pred, val_y)

            train_pred_output = torch.argmax(train_pred, dim=1)
            val_pred_output = torch.argmax(val_pred, dim=1)

            train_report = evaluation_report(
                train_pred_output, data_loader.dataset, dataset
            )
            val_report = evaluation_report(val_pred_output, val_set, dataset)

            train_joint = joint_accuracy(
                train_pred_output, data_loader.dataset, dataset
            )
            val_joint = joint_accuracy(val_pred_output, val_set, dataset)

            reports.append((train_report, val_report, train_joint, val_joint))

            train_accuracy_value = train_report["accuracy"]
            val_accuracy_value = val_report["accuracy"]

            print("batch loss: ", loss.item(), end="\t")
            print("train loss: ", train_loss.item(), end="\t")
            print("val loss: ", val_loss.item(), end="\t")
            print("train accuracy: ", train_accuracy_value, end="\t")
            print("val accuracy: ", val_accuracy_value, end="\t")
            print("train joint accuracy: ", train_joint, end="\t")
            print("val joint accuracy: ", val_joint)
            if is_plot:
                epochs_plot.append(epoch)
                losses_plot.append(loss.item())
                train_losses_plot.append(train_loss.item())
                val_losses_plot.append(val_loss.item())
                train_accuracy_plot.append(train_accuracy_value)
                val_accuracy_plot.append(val_accuracy_value)
                train_joint_plot.append(train_joint)
                val_joint_plot.append(val_joint)
    if is_plot:
        # a figure with 2x1 grid of Axes
        fig, axes = plt.subplots(2, 1, figsize=(10, 10))
        loss_ax, accuracy_ax = axes[0], axes[1]
        loss_ax.set_xlabel("Epoch")
        loss_ax.set_ylabel("Cross Entropy Loss")
        loss_ax.set_title(f"{plot_name} Loss")
        loss_ax.plot(epochs_plot, losses_plot, "g", label="batch loss")
        loss_ax.plot(epochs_plot, train_losses_plot, "b", label="train loss")
        loss_ax.plot(epochs_plot, val_losses_plot, "r", label="val loss")
        loss_ax.legend()
        accuracy_ax.set_xlabel("Epoch")
        accuracy_ax.set_ylabel("Accuracy")
        accuracy_ax.set_ylim((0, 1))
        accuracy_ax.set_title(f"{plot_name} Accuracy")
        accuracy_ax.plot(
            epochs_plot, train_accuracy_plot, "b", label="train accuracy"
        )
        accuracy_ax.plot(
            epochs_plot, val_accuracy_plot, "r", label="val accuracy"
        )
        accuracy_ax.plot(
            epochs_plot, train_joint_plot, "b--", label="train joint accuracy"
        )
        accuracy_ax.plot(
            epochs_plot, val_joint_plot, "r--", label="val joint accuracy"
        )
        accuracy_ax.legend()
        fig.savefig(f"{plot_name}.png")
    return reports
