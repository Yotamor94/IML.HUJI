from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt


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
    raise NotImplementedError()


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        data = np.load(f"../datasets/{f}")
        X = data[:, 0:2]
        y_train = data[:, 2]
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


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        data = np.load(f"../datasets/{f}")
        X = data[:, 0:2]
        y_train = data[:, 2]
        plt.scatter(X[:, 0], X[:, 1], c=y_train)
        plt.show()

        # Fit models and predict over training set
        raise NotImplementedError()

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        from IMLearn.metrics import accuracy
        raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    # compare_gaussian_classifiers()
