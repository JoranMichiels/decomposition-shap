import numpy as np

from decomposition_shap.distributions import MultiGaussian, GaussianCopula


def test_bigaussian():
    np.random.seed(0)
    data = np.random.multivariate_normal([0, 0], np.identity(2), 10000)
    gauss = MultiGaussian(data)
    samps = gauss.sample(features=[1], feature_values=[5], n=10000)
    assert np.allclose(np.mean(samps, axis=0), [0, 5], rtol=0, atol=0.1)
    assert np.allclose(np.cov(samps, rowvar=False), np.array([[1, 0], [0, 0]]), rtol=0, atol=0.1)


def test_trigaussian():
    np.random.seed(0)
    data = np.random.multivariate_normal([0, 0, 0], np.identity(3), 10000)
    gauss = MultiGaussian(data)
    samps = gauss.sample(features=[1, 2], feature_values=[5, 2], n=10000)
    assert np.allclose(np.mean(samps, axis=0), [0, 5, 2], rtol=0, atol=0.1)
    assert np.allclose(np.cov(samps, rowvar=False), np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]]), rtol=0, atol=0.1)

    samps2 = gauss.sample(features=[1], feature_values=[2], n=10000)
    assert np.allclose(np.mean(samps2, axis=0), [0, 2, 0], rtol=0, atol=0.1)
    assert np.allclose(np.cov(samps2, rowvar=False), np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]]), rtol=0, atol=0.1)


def test_dependent():
    np.random.seed(0)
    data = np.random.multivariate_normal([0, 0], np.array([[1, 0.9], [0.9, 1]]), 1000)
    gauss = MultiGaussian(data)
    cond_mean1 = 0.9 * (1 - 0)
    assert np.isclose(cond_mean1, gauss.sample([1], [1], 1, return_moments=True)[1], rtol=0, atol=0.05)


def test_bigaussian_copula():
    np.random.seed(0)
    data = np.random.multivariate_normal([0, 0], np.identity(2), 10000)
    gauss = GaussianCopula(data)
    samps = gauss.sample(features=[1], feature_values=[3], n=10000)
    assert np.allclose(np.mean(samps, axis=0), [0, 3], rtol=0, atol=0.1)
    assert np.allclose(np.cov(samps, rowvar=False), np.array([[1, 0], [0, 0]]), rtol=0, atol=0.1)