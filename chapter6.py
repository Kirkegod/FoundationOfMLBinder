import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps

def load_labeled_mnist_data():
    """Load labeled MNIST data"""

    train_data = np.load("data/chapter6/MNIST_train_data_labeled.npy")
    test_data = np.load("data/chapter6/MNIST_test_data.npy")

    X_train, y_train = train_data[:, :-1], train_data[:, -1]
    X_test, y_test = test_data[:, :-1], test_data[:, -1]

    return X_train, y_train, X_test, y_test
    
def load_full_mnist_data():
    """Load labeled and unlabeled MNIST data"""

    train_data_labeled = np.load("data/chapter6/MNIST_train_data_labeled.npy")
    train_data_unlabeled = np.load("data/chapter6/MNIST_train_data_unlabeled.npy")
    test_data = np.load("data/chapter6/MNIST_test_data.npy")

    X_train_l, y_train_l = train_data_labeled[:, :-1], train_data_labeled[:, -1]
    X_train_u = train_data_unlabeled
    X_test, y_test = test_data[:, :-1], test_data[:, -1]

    return X_train_l, y_train_l, X_train_u, X_test, y_test


def plot_kde(data, ax, ax_lim=None):
    N_plot = 100

    if ax_lim == None:
        data_mean = data.mean(axis=0)
        data_std = data.std(axis=0).max()

        ax_min = data_mean - 2.*data_std
        ax_max = data_mean + 2.*data_std

        ax_lim = (ax_min[0], ax_max[0], ax_min[1], ax_max[1])

    kde = sps.gaussian_kde(data.T)
    x_vals = np.linspace(ax_lim[0], ax_lim[1], N_plot)
    y_vals = np.linspace(ax_lim[3], ax_lim[2], N_plot)

    mesh_coords = np.meshgrid(x_vals, y_vals)
    mesh_coords = np.stack(mesh_coords, 0).reshape(2, -1)

    pdf_est = kde.evaluate(mesh_coords).reshape(N_plot, N_plot)

    im = ax.imshow(pdf_est, cmap=plt.cm.plasma, extent=ax_lim)
    plt.colorbar(im, ax=ax)

def plot_density(f, title=None, save_name=None, n_samples=1000, ax_lim=None):
    z = np.random.randn(n_samples,2)
    x = np.array([f(z_i) for z_i in z])

    plt.rcParams['font.size'] = '20'
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20,10))

    ax[0].set_title(r'$p(\mathbf{z})$', fontsize=25)
    ax[1].set_title(r'$p(\mathbf{x})$', fontsize=25)

    for ax_i, data, ax_i_lim in zip(ax, (z, x), ((-4., 4., -4., 4.), ax_lim)):
        plot_kde(data, ax_i, ax_lim=ax_i_lim)

    if title:
        fig.suptitle(title, fontsize=25)

    if save_name:
        plt.savefig("{}.png".format(save_name))

    fig.patch.set_facecolor("white")

