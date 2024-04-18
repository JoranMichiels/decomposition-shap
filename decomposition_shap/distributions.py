import numpy as np
import statsmodels.api as sm
from scipy import stats


class Distribution(object):
    """
    Defines the distribution class expected in DependentKernelExplainer.
    """

    def sample(self, features, feature_values, n):
        """
        Conditionally sample from the distribution.
        :param features: numpy.array
            Features on which to condition
        :param feature_values: numpy.array
            Values of the former features
        :param n: int
            Number of samples to generate
        :return: The requested samples from the distribution.
        """
        raise NotImplementedError


class MultiGaussian(Distribution):
    """
    Implementation of a multivariate Gaussian distribution which can be sampled conditionally.
    """

    def __init__(self, data=None, mean=None, cov=None):
        """
        Initialize the Gaussian distribution.
        :param data: numpy.array
            Data on which the distribution is fitted. First dimension is the number of samples.
        :param mean: numpy.array (# features)
            If no data is supplied, this the mean of the multivariate Gaussian distribution.
        :param cov: numpy.array (# features, # features)
            If no data is supplied, this the covariance of the multivariate Gaussian distribution.
        """
        if data is not None:
            self.M = data.shape[-1]
            self.mean = np.mean(data, axis=0)
            self.cov = np.cov(data, rowvar=False)
        elif mean is not None and cov is not None:
            self.M = len(mean)
            self.mean = mean
            self.cov = cov
        else:
            raise Exception

    def sample(self, features, feature_values, n, return_moments=False):
        """
        Conditionally sample from the distribution.
        :param features: numpy.array
            Features on which to condition
        :param feature_values: numpy.array
            Values of the former features
        :param n: int
            Number of samples to generate
        :param return_moments: bool
            Whether to also return the conditional mean and covariance
        :return: The requested samples from the distribution. If 'return_moments' is true,
            also the conditional mean and covariance.
        """
        assert len(features) == len(feature_values)

        if len(features) == 0:
            if return_moments:
                return np.random.multivariate_normal(self.mean, self.cov, size=n), self.mean, self.cov
            else:
                return np.random.multivariate_normal(self.mean, self.cov, size=n)

        other_features = [f for f in list(range(self.M)) if f not in features]
        result = np.empty((n, len(self.cov)))
        result[:, features] = feature_values
        if len(features) == 1:
            inv = 1 / self.cov[features, features]
            if len(other_features) == 1:
                cond_mean = self.mean[other_features] + inv * self.cov[other_features, features] * (
                        feature_values - self.mean[features])
                cond_variance = self.cov[other_features, other_features] - self.cov[other_features, features] * inv * \
                                self.cov[features, other_features]
                result[:, other_features] = np.random.normal(cond_mean, cond_variance, size=n).reshape((-1, 1))
            else:
                cond_mean = self.mean[other_features] + inv * self.cov[np.ix_(other_features, features)] @ (
                        feature_values - self.mean[features])
                cond_variance = self.cov[np.ix_(other_features, other_features)] - inv * self.cov[
                    np.ix_(other_features, features)] @ self.cov[np.ix_(features, other_features)]
                result[:, other_features] = np.random.multivariate_normal(cond_mean, cond_variance, size=n)
        else:
            inv = np.linalg.inv(self.cov[np.ix_(features, features)])
            cond_mean = self.mean[other_features] + self.cov[np.ix_(other_features, features)] @ inv @ (
                    feature_values - self.mean[features])
            cond_variance = self.cov[np.ix_(other_features, other_features)] - self.cov[
                np.ix_(other_features, features)] @ inv @ self.cov[np.ix_(features, other_features)]
            result[:, other_features] = np.random.multivariate_normal(cond_mean, cond_variance, size=n)
        if return_moments:
            return result, cond_mean, cond_variance
        else:
            return result


class GaussianCopula(Distribution):
    """
    Implementation of a Gaussian Copula distribution which can be sampled conditionally.
    """

    def __init__(self, data, cov=None):
        """
        Initialize the Copula distribution.
        :param data: numpy.array
            Data on which the distribution is fitted. First dimension is the number of samples.
        :param cov: numpy.array (# features, # features)
            Enforce the Gaussian Copula to have this covariance, otherwise it is computed from the data.
        """
        self.M = data.shape[1]

        # Compute empirical cumulative distributions for marginals
        self.ecdfs = [ECDF(data[:, i]) for i in range(self.M)]
        self.inverse_ecdfs = [InverseECDF(data[:, i]) for i in range(self.M)]

        # Compute transformed distribution
        if cov is None:
            cumulative = data.copy()
            for i in range(self.M):
                cumulative[:, i] = self.ecdfs[i](data[:, i])

            cov = np.cov(stats.norm.ppf(cumulative), rowvar=False)

        self.multi_gaussian = MultiGaussian(mean=np.zeros(self.M), cov=cov)

    def sample(self, features, feature_values, n):
        # Transform variables
        result = self.multi_gaussian.sample(features, stats.norm.ppf(
            [self.ecdfs[features[i]](feature_values[i]) for i in range(len(features))]), n)

        result = stats.norm.cdf(result)

        for i in range(self.M):
            result[:, i] = self.inverse_ecdfs[i](result[:, i])

        return result


class InverseECDF(object):
    def __init__(self, x):
        self.sorted_x = np.sort(x)

    def __call__(self, p):
        # Find datapoint bigger than p of all other data points
        return self.sorted_x[(len(self.sorted_x) * p).astype(int)]


class ECDF(object):
    def __init__(self, x, eps=1e-15):
        self.eps = eps
        self.sm_ecdf = sm.distributions.empirical_distribution.ECDF(x)

    def __call__(self, x):
        vals = np.array([self.sm_ecdf(x)])  # because: can be value and array
        vals[vals == 0] = self.eps
        vals[vals == 1] = 1 - self.eps
        return vals[0]
