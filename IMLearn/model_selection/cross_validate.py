from __future__ import annotations

from typing import Tuple, Callable

import numpy as np

from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_samples)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    n_samples = X.shape[0]
    test_size = int(np.ceil((1 / cv) * n_samples))
    train_results = np.zeros(cv)
    test_results = np.zeros(cv)
    for iter in range(cv):
        train_X = np.concatenate([X[0: iter * test_size], X[test_size * (iter + 1): n_samples]], 0)
        train_y = np.concatenate([y[0: iter * test_size], y[test_size * (iter + 1): n_samples]], 0)
        test_X = X[iter * test_size: (iter + 1) * test_size]
        test_y = y[iter * test_size: (iter + 1) * test_size]
        estimator.fit(train_X, train_y)
        train_results[iter] = scoring(estimator.predict(train_X), train_y)
        test_results[iter] = scoring(estimator.predict(test_X), test_y)
    return train_results.mean(), test_results.mean()
