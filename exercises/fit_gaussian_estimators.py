import matplotlib.pyplot as plt
import numpy as np
import plotly.io as pio

from IMLearn.learners import UnivariateGaussian, MultivariateGaussian

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    SAMPLES = 1000
    MEAN = 10
    VARIANCE = 1
    # Question 1 - Draw samples and print fitted model
    samples = np.random.normal(MEAN, VARIANCE, SAMPLES)
    univariate_estimator = UnivariateGaussian()
    univariate_estimator.fit(samples)
    print('(', univariate_estimator.mu_, ', ', univariate_estimator.var_, ')', sep='')

    # Question 2 - Empirically showing sample mean is consistent
    sample_sizes = np.arange(10, SAMPLES, 10)
    results = []
    for size in sample_sizes:
        univariate_estimator = UnivariateGaussian()
        univariate_estimator.fit(samples[0:size])
        results.append(np.abs(univariate_estimator.mu_ - MEAN))
    plt.plot(sample_sizes, results)
    plt.title("Accuracy of the Estimator")
    plt.xlabel("number of samples")
    plt.ylabel("distance from actual mean")
    plt.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    # In the plot I expect to see a univariate gaussian distribution function
    samples.sort()
    pdf = univariate_estimator.pdf(samples)
    plt.scatter(samples, pdf, s=0.1)
    plt.title("pdf of Drawn Samples According to Estimated Gaussian")
    plt.xlabel("sample value")
    plt.ylabel("pdf value")
    plt.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = [0, 0, 4, 0]
    sigma = [[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]]
    samples = np.random.multivariate_normal(mu, sigma, 1000)
    multivariate_estimator = MultivariateGaussian()
    multivariate_estimator.fit(samples)
    print(multivariate_estimator.mu_)
    print(multivariate_estimator.cov_)

    # Question 5 - Likelihood evaluation
    SIZE = 200
    f1 = np.linspace(-10, 10, SIZE)
    f3 = np.linspace(-10, 10, SIZE)
    res = np.ndarray((SIZE, SIZE))
    for i, val1 in enumerate(f1):
        for j, val3 in enumerate(f3):
            mu = [val1, 0, val3, 0]
            log_likelihood = MultivariateGaussian.log_likelihood(np.array(mu).transpose(), np.array(sigma), samples)
            res[i, j] = log_likelihood
        print(val1, i)
    plt.imshow(res, cmap='viridis', extent=[-10, 10, -10, 10])
    plt.colorbar()
    plt.title('log likelihood as a function of the different mean values')
    plt.xlabel('f3')
    plt.ylabel('f1')
    plt.show()

    # Question 6 - Maximum likelihood
    max_ind = np.unravel_index(res.argmax(), res.shape)
    print("max likelihood coordinates =", max_ind, f'values are ({f1[max_ind[0]]}, {f3[max_ind[1]]})')
    print("max likelihood value =", res[max_ind])


if __name__ == '__main__':
    np.random.seed(0)
    # data = np.array([1, 5, 2, 3, 8, -4, -2, 5, 1, 10, -10, 4, 5, 2, 7, 1, 1, 3, 2, -1, -3, 1, -4, 1, 2, 1,
    #                  -4, -4, 1, 3, 2, 6, -6, 8, 3, -6, 4, 1, -2, 3, 1, 4, 1, 4, -2, 3, -1, 0, 3, 5, 0, -2])
    # print(UnivariateGaussian.log_likelihood(10, 1, data))
    # test_univariate_gaussian()
    test_multivariate_gaussian()
