from typing import Tuple

from IMLearn.learners.classifiers import DecisionStump
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.metrics import accuracy
from utils import *


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    adaboost = AdaBoost(lambda: DecisionStump(), n_learners)
    adaboost.fit(train_X, train_y)

    n_learners_x = np.linspace(1, n_learners, n_learners).astype(int)

    test_loss = np.zeros(n_learners)
    train_loss = np.zeros(n_learners)
    for n in n_learners_x:
        train_loss[n - 1] = adaboost.partial_loss(train_X, train_y, n)
        test_loss[n - 1] = adaboost.partial_loss(test_X, test_y, n)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=n_learners_x, y=train_loss, name="train"))
    fig.add_trace(go.Scatter(x=n_learners_x, y=test_loss, name="test"))
    fig.update_layout(title="Loss as a Function of Number of Models")
    fig.show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([(-1, 1), (-1, 1)])
    fig = make_subplots(2, 2, subplot_titles=[f"Decision Boundary With {t} Models" for t in T])
    for i, t in enumerate(T):
        decision_surf = decision_surface(lambda X: adaboost.partial_predict(X, t), lims[0], lims[1], showscale=False)
        scatter = go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", marker=dict(color=test_y))
        fig.add_traces([decision_surf, scatter], i // 2 + 1, i % 2 + 1)
    fig.update_layout(showlegend=False)
    fig.show()

    # Question 3: Decision surface of best performing ensemble
    best_num = n_learners_x[test_loss.argmin()]
    fig = go.Figure()
    decision_surf = decision_surface(lambda X: adaboost.partial_predict(X, best_num), lims[0], lims[1], showscale=False)
    scatter = go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", marker=dict(color=test_y))
    fig.add_traces([decision_surf, scatter])
    fig.update_layout(showlegend=False,
                      title=f"Decision Boundary with {best_num} Models, Accuracy = {accuracy(test_y, adaboost.partial_predict(test_X, best_num))}")
    fig.show()
    # Question 4: Decision surface with weighted samples
    fig = go.Figure()
    decision_surf = decision_surface(adaboost.predict, lims[0], lims[1], showscale=False)
    D = adaboost.D_
    D = D / D.max() * 15
    scatter = go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", marker=dict(color=train_y, size=D))
    fig.add_traces([decision_surf, scatter])
    fig.update_layout(showlegend=False, title=f"Train Data with Color Indicating Class and Size indicating D[T]")
    fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)
