
from ..utils.tools import Solver

import numpy as np
import copy

class EM(Solver):
    def __init__(self, max_iter=100, theta=1e-5):
        self.max_iter = max_iter
        self.theta = theta

    def _init_parameters(self, X):
        rows, cols = X.shape
        mu_init = np.nanmean(X, axis=0)
        sigma_init = np.zeros((cols, cols))
        for i in range(cols):
            for j in range(i, cols):
                vec_col = X[:, [i, j]]
                vec_col = vec_col[~np.any(np.isnan(vec_col), axis=1), :].T
                if len(vec_col) > 0:
                    cov = np.cov(vec_col)
                    cov = cov[0, 1]
                    sigma_init[i, j] = cov
                    sigma_init[j, i] = cov

                else:
                    sigma_init[i, j] = 1.0
                    sigma_init[j, i] = 1.0

        return mu_init, sigma_init

    def _e_step(self, mu,sigma, X):
        samples,_ = X.shape
        for sample in range(samples):
            if np.any(np.isnan(X[sample,:])):
                loc_nan = np.isnan(X[sample,:])
                new_mu = np.dot(sigma[loc_nan, :][:, ~loc_nan],
                                np.dot(np.linalg.inv(sigma[~loc_nan, :][:, ~loc_nan]),
                                       (X[sample, ~loc_nan] - mu[~loc_nan])[:,np.newaxis]))
                nan_count = np.sum(loc_nan)
                X[sample, loc_nan] = mu[loc_nan] + new_mu.reshape(1,nan_count)

        return X

    def _m_step(self,X):
        rows, cols = X.shape
        mu = np.mean(X, axis=0)
        sigma = np.cov(X.T)
        tmp_theta = -0.5 * rows * (cols * np.log(2 * np.pi) +
                                  np.log(np.linalg.det(sigma)))

        return mu, sigma,tmp_theta



    def solve(self, X):
        mu, sigma = self._init_parameters(X)
        complete_X,updated_X = None, None
        rows,_ = X.shape
        theta = -np.inf
        for iter in range(self.max_iter):
            updated_X = self._e_step(mu=mu, sigma=sigma, X=copy.copy(X))
            mu, sigma, tmp_theta = self._m_step(updated_X)
            for i in range(rows):
                tmp_theta -= 0.5 * np.dot((updated_X[i, :] - mu),
                                          np.dot(np.linalg.inv(sigma), (updated_X[i, :] - mu)[:, np.newaxis]))
            if abs(tmp_theta-theta)<self.theta:
                complete_X = updated_X
                break;
            else:
                theta = tmp_theta
        else:
            complete_X = updated_X
        return complete_X