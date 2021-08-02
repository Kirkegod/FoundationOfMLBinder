import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import scipy.stats as sps
from sklearn.decomposition import PCA

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

def load_full_reduced_mnist_data():
    """Load labeled and unlabeled MNIST data"""

    train_data_labeled = np.load("data/chapter6/MNIST_train_data_labeled.npy")
    train_data_unlabeled = np.load("data/chapter6/MNIST_train_data_unlabeled.npy")
    test_data = np.load("data/chapter6/MNIST_test_data.npy")

    X_train_l, y_train_l = train_data_labeled[:, :-1], train_data_labeled[:, -1]
    X_train_u = train_data_unlabeled
    X_train_full = np.concatenate((X_train_l, X_train_u))

    X_test, y_test = test_data[:, :-1], test_data[:, -1]

    # Perform PCA (based on full train set)
    pca = PCA(n_components=2) # Leave empty
    X_train_full_reduced = pca.fit_transform(X_train_full)
    X_test_reduced = pca.transform(X_test)

    # Split the data again into labeled and unlabeled sets
    X_train_l_reduced = X_train_full_reduced[:X_train_l.shape[0], :]
    X_train_u_reduced = X_train_full_reduced[X_train_l.shape[0]:, :]

    return X_train_l_reduced, y_train_l, X_train_u_reduced, X_test_reduced, y_test

def visualize_data(X_l, y_l, X_u=None, ax=None, show=True):
    """Visualise 2D class data
    X_l: (n_l, p) labeled input
    y_l: (n_l,) targets for labeled input
    X_u: (n_u, p) unlabeled input
    """

    if ax is None:
        fig, ax = plt.subplots()

    if X_u is not None:
        ax.scatter(X_u[:, 0], X_u[:, 1], c='#7f7f7f', alpha=0.5)

    classes = ['6', '7', '8']
    for i in range(len(classes)):
        ax.scatter(X_l[y_l == i, 0], X_l[y_l == i, 1])

    if X_u is not None:
        classes = ['unknown'] + classes

    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend(classes)

    if show:
        plt.show()

def visualize_clusters(X_train, y_train, mu, sigma, X_train_u=None):
    """Visualize clusters
    X_train: (n_train, 2) input array of dimension 2
    y_train: (n_train,) target array
    mu: (n_classes, 2) array of estimated means
    sigma: (n_classes, 2, 2) array of estimated covariance matrices
    X_train_u: (n_train_u, 2) additional unlabeled data
    """

    # Plot data
    fig, ax = plt.subplots()

    visualize_data(X_train, y_train, X_train_u, ax, show=False)

    classes = np.unique(y_train)
    for i in range(len(classes)):
        # Plot distribution contour
        draw_ellipse(mu[i, :], sigma[i, :, :], ax)

    plt.show()

def draw_ellipse(mu, sigma, ax):
    """Draw 2D ellipse corresponding to contour of Gaussian distribution with specified mean and covariance (2 stds)
    mu: (2,) mean vector
    sigma: (2, 2) covariance matrix
    ax: plot axis
    """

    # Convert covariance to principal axes
    u, s, vh = np.linalg.svd(sigma)
    angle = float(np.degrees(np.arctan2(u[1, 0], u[0, 0])))
    width, height = 2 * np.sqrt(s)

    # Draw ellipse
    ax.add_patch(Ellipse(mu, 2 * width, 2 * height, angle, edgecolor='k', facecolor='none'))


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

