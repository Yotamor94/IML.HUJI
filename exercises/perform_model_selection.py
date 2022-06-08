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
        train_results[degree], val_results[degree] = cross_validate(PolynomialFitting(degree), train_x, train_y,
                                                                    mean_square_error)

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
    print("Test Error =", np.round(mean_square_error(val_y, polyfit.predict(val_x)), 2), "Validation Error =",
          val_results[best_k])


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
    X, y = datasets.load_diabetes(return_X_y=True, as_frame=True)
    train_x, train_y, test_x, test_y = split_train_test(X, y, 50 / 442)
    train_x = train_x.to_numpy()
    train_y = train_y.to_numpy()
    test_x = test_x.to_numpy()
    test_y = test_y.to_numpy()

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    ridge_train_results, ridge_val_results = np.zeros(n_evaluations), np.zeros(n_evaluations)
    lasso_train_results, lasso_val_results = np.zeros(n_evaluations), np.zeros(n_evaluations)
    lambdas = np.linspace(0.012, 1, n_evaluations)
    print(train_x.shape)
    for i, lam in enumerate(lambdas):
        ridge_train_results[i], ridge_val_results[i] = cross_validate(RidgeRegression(lam), train_x, train_y,
                                                                      mean_square_error)
        lasso_train_results[i], lasso_val_results[i] = cross_validate(Lasso(lam), train_x, train_y, mean_square_error)

    ridge_trace_train_scores = go.Scatter(x=lambdas, y=ridge_train_results, name="Ridge RegressionTrain MSE")
    lasso_trace_train_scores = go.Scatter(x=lambdas, y=lasso_train_results, name="Lasso Regression Train MSE")
    ridge_trace_val_scores = go.Scatter(x=lambdas, y=ridge_val_results, name="Ridge Regression Validation MSE")
    lasso_trace_val_scores = go.Scatter(x=lambdas, y=lasso_val_results, name="Lasso Regression Validation MSE")
    layout = go.Layout(title="MSE as a Function of lambda", xaxis=dict(title="lambda"), yaxis=dict(title="MSE"))
    fig = go.Figure(
        [ridge_trace_train_scores, ridge_trace_val_scores, lasso_trace_train_scores, lasso_trace_val_scores], layout)
    fig.show()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    best_lam_ridge = lambdas[ridge_val_results.argmin()]
    best_lam_lasso = lambdas[lasso_val_results.argmin()]
    ridge = RidgeRegression(best_lam_ridge)
    lasso = Lasso(best_lam_lasso)
    ls = LinearRegression()
    ridge.fit(train_x, train_y)
    lasso.fit(train_x, train_y)
    ls.fit(train_x, train_y)
    print("Best lambda for Ridge =", best_lam_ridge)
    print("Best lambda for Lasso =", best_lam_lasso)
    print("Ridge Test Error =", np.round(mean_square_error(test_y, ridge.predict(test_x)), 2), "Validation Error =", np.round(ridge_val_results.min(initial=100000), 2))
    print("Lasso Test Error =", np.round(mean_square_error(test_y, lasso.predict(test_x)), 2), "Validation Error =", np.round(lasso_val_results.min(initial=100000), 2))
    print("Least Squares Test Error =", np.round(mean_square_error(test_y, ls.predict(test_x)), 2))

if __name__ == '__main__':
    np.random.seed(0)
    # select_polynomial_degree()
    select_polynomial_degree(noise=0)
    # select_polynomial_degree(1500, 10)
    np.random.seed(0)
    # select_regularization_parameter()
