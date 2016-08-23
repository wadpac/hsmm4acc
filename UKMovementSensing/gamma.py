from __future__ import division
from past.utils import old_div
import pybasicbayes
import numpy as np
from scipy import special
from pybasicbayes.abstractions import GibbsSampling, MeanField, \
    MeanFieldSVI, Collapsed, MaxLikelihood, MAP


class GammaFixedRate(
    GibbsSampling, MeanField, MeanFieldSVI,
    Collapsed, MAP, MaxLikelihood):


    def __init__(self, alpha=None, beta=None, a_0=None, b_0=None, c_0=None):
        self.alpha = alpha
        self.beta = beta
        self.a_0 = a_0
        self.b_0 = b_0
        self.c_0 = c_0
        # TODO: set if they are none

    def max_likelihood(self, data, weights=None):
        super(GammaFixedRate, self).max_likelihood(data, weights)

    def log_marginal_likelihood(self, data):
        super(GammaFixedRate, self).log_marginal_likelihood(data)

    def MAP(self, data, weights=None):
        super(GammaFixedRate, self).MAP(data, weights)

    def _get_statistics(self, data):
        if isinstance(data, np.ndarray):
            n = data.shape[0]
            prod = data.prod()
        elif isinstance(data, list):
            n = sum(d.shape[0] for d in data)
            prod = np.prod(d.prod() for d in data)
        else:
            assert np.isscalar(data)
            n = 1
            prod = data

        return n, prod

    @property
    def params(self):
        return dict(alpha=self.alpha, beta=self.beta)

    @property
    def hypparams(self):
        return dict(a_0=self.a_0, b_0=self.b_0, c_0=self.c_0)

    def resample(self, data=[]):
        n, prod = self._get_statistics(data)
        self.a_0 *= prod
        self.b_0 +=  n
        self.c_0 +=  n
        self.alpha = self._sample_alpha(n=1)[0]

    def _sample_alpha(self, n=1):
        eps = 1e-5
        loglikelihood = lambda alpha: (alpha - 1) * np.log(self.a_0+eps) + alpha * self.c_0 * np.log(self.beta+eps) \
                                      - self.b_0 * np.log(special.gamma(alpha))
        likelihood = lambda alpha: np.exp(loglikelihood(alpha))
        stop = 15
        alpha_space = np.linspace(0, stop, num=old_div(stop, 0.001))
        alpha_dist = likelihood(alpha_space)
        alpha_dist = old_div(alpha_dist, alpha_dist.sum())
        return np.random.choice(a=alpha_space, p=alpha_dist, size=n)

    def expected_log_likelihood(self, x):
        super(GammaFixedRate, self).expected_log_likelihood(x)

    def meanfield_sgdstep(self, expected_suff_stats, prob, stepsize):
        super(GammaFixedRate, self).meanfield_sgdstep(expected_suff_stats, prob, stepsize)

    def meanfieldupdate(self, data, weights):
        super(GammaFixedRate, self).meanfieldupdate(data, weights)

    def copy_sample(self):
        return super(GammaFixedRate, self).copy_sample()

    def rvs(self, size=[]):
        super(GammaFixedRate, self).rvs(size)

    @property
    def num_parameters(self):
        pass

    def log_likelihood(self, x):
        super(GammaFixedRate, self).log_likelihood(x)

    def get_vlb(self):
        pass