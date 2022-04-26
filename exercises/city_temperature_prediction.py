import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test


def get_day_of_year(year: np.ndarray, month: np.ndarray, day: np.ndarray):
    N1 = 275 * month // 9
    N2 = (month + 9) // 12
    N3 = (1 + (year - 4 * year // 4 + 2) // 3)
    return N1 - (N2 * N3) + day - 30


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=True)
    df = df.loc[df.Temp > -10]
    df['DayOfYear'] = get_day_of_year(df.Year, df.Month, df.Day)
    df = df.reset_index()
    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data('../datasets/City_Temperature.csv')

    # Question 2 - Exploring data for specific country
    df.loc[df.Country == 'Israel'].plot.scatter('DayOfYear', 'Temp', c=df[df.Country == 'Israel'].Year, cmap='Set3')
    plt.title('Temperature as a Function of the Day of the Year')
    plt.tight_layout()
    plt.show()

    # Question 3 - Exploring differences between countries
    group = df[df.Country == 'Israel'].groupby('Month')
    group.agg('std')['Temp'].plot.bar()
    plt.title('Temperature Standard deviation by Month')
    plt.show()
    group.agg('mean')['Temp'].plot.bar(yerr=group.agg('std')['Temp'], capsize=6)
    plt.grid()
    plt.title('Mean Temperature by Month with Standard Deviation')
    plt.show()
    group2 = df.groupby(['Country', 'Month'])
    mean_df = group2.agg({'Temp': ['mean', 'std']}).unstack(level=0)
    mean_df['Temp', 'mean'].plot(yerr=mean_df['Temp', 'std'])
    plt.title('Average Temperature per Month per Country')
    plt.show()

    # Question 4 - Fitting model for different values of `k`
    Isr = df[df.Country == 'Israel'].reset_index()
    X_train, y_train, X_test, y_test = split_train_test(Isr['DayOfYear'], Isr['Temp'])
    losses = []
    for k in range(1, 11):
        poly_model = PolynomialFitting(k)
        poly_model.fit(X_train.to_numpy(), y_train.to_numpy())
        losses.append(poly_model.loss(X_test.to_numpy(), y_test.to_numpy()))
        print(f'{k}: {round(losses[-1], 2)}')

    plt.bar(range(1, 11), losses)
    plt.title('Loss of the Model on the Samples from Israel\nas a Function of the Polynom Degree')
    plt.show()

    # Question 5 - Evaluating fitted model on different countries
    poly_model = PolynomialFitting(5)
    poly_model.fit(X_train.to_numpy(), y_train.to_numpy())
    errors = []
    for country in df.Country.unique():
        country_df = df[df['Country'] == country]
        errors.append(poly_model.loss(country_df['DayOfYear'], country_df['Temp']))
        plt.scatter(country_df['DayOfYear'], country_df['Temp'])
        plt.scatter(country_df['DayOfYear'], poly_model.predict(country_df['DayOfYear']), c='r')
        plt.title(country)
        plt.show()
        print(f'{country}: {errors[-1]}')

    plt.bar(df.Country.unique(), errors)
    plt.title('Mean Squared Error of Model as\n a Function of the Country')
    plt.show()
