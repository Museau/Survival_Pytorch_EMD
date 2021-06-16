from matplotlib import pyplot as plt


def plot_train_val_history(path_fig, fig_name, train_hist, val_hist):
    """
    Plot training and validation history.

    Parameters
    ----------
    path_fig : str
        Figure path.
    fig_name : str
        Figure name.
    train_hist : list
        Training history.
    val_hist : list
        Validation history.
    """
    epoch = len(train_hist)
    x = range(1, epoch + 1)
    plt.plot(x, train_hist, label="train")
    plt.plot(x, val_hist, label="val")
    plt.legend()
    plt.savefig(f"{path_fig}{fig_name}.png")
    plt.close()
