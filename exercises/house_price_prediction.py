from typing import NoReturn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from IMLearn.learners.regressors import LinearRegression
from IMLearn.utils import split_train_test


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv(filename)
    df = df.loc[df.price > 0]
    df = df.loc[df.sqft_lot15 > 0]
    df = df.loc[df.zipcode != 0]
    df.loc[df.yr_renovated == 0, 'yr_renovated'] = df.yr_built
    df = df.reset_index()
    prices = df.price
    df = pd.concat([df, pd.get_dummies(df.zipcode)], axis=1)
    df = df.drop(['id', 'date', 'zipcode', 'lat', 'long', 'price'], axis=1)
    return df, prices


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    for feature in X.columns:
        pearson_correlation = X[feature].cov(y) / (y.std() * X[feature].std())
        plt.scatter(X[feature], y, s=0.4, c='r')
        plt.title(
            f'Response as a Function of {feature}\n pearson correlation between them: {round(pearson_correlation, 3)}\n')
        plt.xlabel(feature)
        plt.ylabel('response')
        plt.savefig(f'{output_path}/{feature}.png', bbox_inches='tight')


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    # load_data('../datasets/house_prices.csv')
    X, y = load_data('../datasets/house_prices.csv')

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X, y, './feature_graphs')

    # Question 3 - Split samples into training- and testing sets.
    X_train, y_train, X_test, y_test = split_train_test(X, y)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    n_samples_train = X_train.shape[0]
    regressor = LinearRegression()
    percentages = range(10, 100, 1)
    losses = []
    stds = []
    for p in percentages:
        p = p / 100
        inner_losses = np.zeros(10)
        print(p)
        for i in range(10):
            randstate = np.random.randint(0, 100000)
            sample = X_train.sample(frac=p, random_state=randstate)
            sample_response = y_train.sample(frac=p, random_state=randstate)
            regressor.fit(sample.to_numpy(), sample_response.to_numpy())
            inner_losses[i] = regressor.loss(X_test.to_numpy(), y_test.to_numpy())
        losses.append(inner_losses.mean())
        stds.append(inner_losses.std())

    losses = np.array(losses)
    ci = 2 * np.array(stds)
    plt.plot(percentages, losses)
    plt.fill_between(percentages, (losses - ci), (losses + ci), color='blue', alpha=0.1)
    plt.show()
