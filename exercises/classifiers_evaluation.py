from math import atan2, pi
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from IMLearn.learners.classifiers import LDA, GaussianNaiveBayes
from IMLearn.learners.classifiers import Perceptron
from IMLearn.metrics import accuracy


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class
    Parameters
    ----------
    filename: str
        Path to .npy data file
    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used
    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class
    """
    data = np.load("../datasets/" + filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y_train = load_dataset(f)
        plt.scatter(X[:, 0], X[:, 1], c=y_train)
        plt.show()

        # Fit Perceptron and record loss in each fit iteration
        losses = []

        def get_losses_callback(fit: Perceptron, x: np.ndarray, y: int):
            losses.append(fit.loss(X, y_train))

        perceptron = Perceptron(callback=get_losses_callback)
        perceptron.fit(X, y_train)

        # Plot figure
        plt.plot(losses)
        plt.title(f"Perceptron Loss on {n} data")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix
    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse
    cov: ndarray of shape (2,2)
        Covariance of Gaussian
    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black", showlegend=False)


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y_train = load_dataset(f)
        plt.scatter(X[:, 0], X[:, 1], c=y_train)
        plt.show()

        # Fit models and predict over training set
        lda = LDA()
        lda.fit(X, y_train)
        gaussian_naive_bayes = GaussianNaiveBayes()
        gaussian_naive_bayes.fit(X, y_train)
        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        lda_pred = lda.predict(X)
        lda_mu = lda.mu_
        lda_cov = lda.cov_
        gaussian_pred = gaussian_naive_bayes.predict(X)
        gaussian_mu = gaussian_naive_bayes.mu_
        gaussian_cov = gaussian_naive_bayes.vars_
        fig = make_subplots(1, 2, subplot_titles=(
            f"Gaussian Naive Bayes:\nAccuracy = {accuracy(y_train, gaussian_pred):0.3f}",
            f"LDA:\nAccuracy = {accuracy(y_train, lda_pred):0.3f}"))
        fig.add_trace(
            go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers",
                       marker=dict(color=gaussian_pred, symbol=y_train, size=8, line=dict(width=1,
                                                                                          color='SlateGrey')),
                       showlegend=False
                       ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers",
                       marker=dict(color=lda_pred, symbol=y_train, size=8, line=dict(width=1,
                                                                                     color='SlateGrey')),
                       showlegend=False),
            row=1, col=2
        )
        # Add traces for data-points setting symbols and colors

        # Add `X` dots specifying fitted Gaussians' means
        fig.add_trace(
            go.Scatter(x=lda_mu[:, 0], y=lda_mu[:, 1], mode="markers", marker_symbol='x',
                       marker=dict(size=12, color='black'), showlegend=False),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=gaussian_mu[:, 0], y=gaussian_mu[:, 1], mode="markers", marker_symbol='x',
                       marker=dict(size=12, color='black'), showlegend=False),
            row=1, col=1
        )

        # Add ellipses depicting the covariances of the fitted Gaussians
        cov_gaus = np.zeros((2, 2))
        for i in range(lda_mu.shape[0]):
            ellipse_lda = get_ellipse(lda_mu[i], lda_cov)
            fig.add_trace(ellipse_lda, row=1, col=2)
            np.fill_diagonal(cov_gaus, gaussian_cov[i])
            ellipse_lda = get_ellipse(gaussian_mu[i], cov_gaus)
            fig.add_trace(ellipse_lda, row=1, col=1)

        fig.update_layout(title_text="Classification (color) and True Value (shape) for different models")
        fig.show()

if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
