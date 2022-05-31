from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    x = np.linspace(-1.2, 2, n_samples)
    f_x_no_noise = (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)
    gaussian_noise = np.random.normal(0, np.sqrt(noise), n_samples)
    f_x_noise = f_x_no_noise + gaussian_noise
    train_x, train_y, val_x, val_y = split_train_test(pd.DataFrame(x), pd.Series(f_x_noise), 2 / 3)
    train_x = train_x.to_numpy().flatten()
    train_y = train_y.to_numpy()
    val_x = val_x.to_numpy().flatten()
    val_y = val_y.to_numpy()

    trace_f_x_no_noise = go.Scatter(x=x, y=f_x_no_noise, marker=dict(size=3), name="f(x) with no noise")
    trace_train = go.Scatter(x=train_x, y=train_y, mode="markers", name="train")
    trace_validation = go.Scatter(x=val_x, y=val_y, mode="markers", name="validation")
    layout = go.Layout(title="f(x) Next to Train and Test Data", xaxis=dict(title="x"), yaxis=dict(title="f(x)"))
    fig = go.Figure([trace_f_x_no_noise, trace_train, trace_validation], layout)
    fig.show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    train_results, val_results = np.zeros(11), np.zeros(11)
    degrees = list(range(11))
    for degree in degrees:
        train_results[degree], val_results[degree] = cross_validate(PolynomialFitting(degree), train_x, train_y, mean_square_error)

    trace_train_scores = go.Scatter(x=degrees, y=train_results, name="Train MSE")
    trace_val_scores = go.Scatter(x=degrees, y=val_results, name="Validation MSE")
    layout = go.Layout(title="MSE as a Function of k", xaxis=dict(title="k"), yaxis=dict(title="MSE"))
    fig = go.Figure([trace_train_scores, trace_val_scores], layout)
    fig.show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    best_k = val_results.argmin()
    polyfit = PolynomialFitting(best_k)
    polyfit.fit(train_x, train_y)
    print("Best k =", best_k)
    print("Test Error =", np.round(mean_square_error(val_y, polyfit.predict(val_x)), 2), "Validation Error =", val_results[best_k])




def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    raise NotImplementedError()

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    raise NotImplementedError()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(noise=0)
    select_polynomial_degree(1500, 10)