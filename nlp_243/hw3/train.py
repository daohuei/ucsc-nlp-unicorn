from matplotlib import pyplot as plt

from evaluate import val_accuracy


def train(
    model,
    n_epochs,
    data_loader,
    loss_func,
    optimizer,
    val_set,
    dataset,
    is_plot=False,
):
    epochs_plot = []
    losses_plot = []

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
            print(
                "val accuracy: ",
                val_accuracy(model, val_set, dataset),
                end="\t",
            )
            print("loss: ", loss.item())
            if is_plot:
                epochs_plot.append(epoch)
                losses_plot.append(loss.item())
    if is_plot:
        # a figure with 2x1 grid of Axes
        fig, ax = plt.subplots(figsize=(10, 10))

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cross Entropy Loss")
        ax.set_title("Baseline Performance")
        _ = ax.plot(epochs_plot, losses_plot)
