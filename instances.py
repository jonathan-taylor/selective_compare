from __future__ import division, print_function
import numpy as np

_cholesky_factors = {}

class highdim_knockoff(object):

    def __init__(self, 
                 n=2000,
                 p=2500,
                 k=30,
                 rho=0.,
                 signal=4.5,
                 sigma=2.):

        (self.n,
         self.p,
         self.k,
         self.rho,
         self.signal,
         self.sigma) = (n, p, k, rho, signal, sigma)

    @property
    def cov(self):
        if not hasattr(self, "_cov"):
            idx = np.arange(self.p)
            self._cov = self.rho**(np.subtract.outer(idx, idx))
        return self._cov

    @property
    def cholesky(self, doc='lower triangular cholesky factorization'):
        if not hasattr(self, "_cholesky"):
            if (self.rho, self.p) in _cholesky_factors.keys():
                _cholesky_factors[(self.rho, self.p)] = np.linalg.cholesky(self.cov)
            self._cholesky = _cholesky_factors[(self.rho, self.p)]
        return self._cholesky

    def instance(self):
        """
        Generate a random instance.
        """
        n, p, k = self.n, self.p, self.k
        X = np.random.standard_normal((n, p)).dot(_sqrt_cov.T)
        X /= (np.sqrt((X**2).sum(0)))[None,:] # normalize columns to have length 1
        beta = np.zeros(p)
        beta[:k] = self.signal * (2 * np.random.binomial(1, 0.5, size=(k,)) - 1) 
        np.random.shuffle(beta)
        Y = (X.dot(beta) + np.random.standard_normal(n)) * sigma
        return X, Y, beta

    def evaluate(self, X, Y, beta, result):

        

    
