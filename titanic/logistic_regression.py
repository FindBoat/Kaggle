#!/usr/bin/env python
from numpy import *
from scipy.optimize import fmin_bfgs

class LogisticRegression:
    """ An implementation of logistic regression. """
    def __init__ (self, x, y, lambda_=0):
        self.x = x
        self.y = atleast_2d(y).transpose()
        self._lambda = lambda_

    def _sigmoid(self, x):
        res = 1 / (1 + exp(-x))
        idx = res == 1
        res[idx] = .99
        return res

    def _compute_cost(self, theta):
        """ Calculate the cost function:
        J = -1 / m * (y' * log(sigmoid(X * theta)) + (1 .- y') * log(1 .- sigmoid(X * theta)))
        J += lambda / (2 * m) * theta(2 : end)' * theta(2 : end)
        """
        m = self.x.shape[0]
        x_bias = hstack((ones((m, 1)), self.x))
        theta = atleast_2d(theta).transpose()
        J = -1.0 / m * (dot(self.y.transpose(), log(self._sigmoid(dot(x_bias, theta))))
            + dot(1 - self.y.transpose(), log(1 - self._sigmoid(dot(x_bias, theta)))))
        J += self._lambda / (2 * m) * sum(theta[1 : :] ** 2)
        return J[0, 0]

    def _compute_grad(self, theta):
        """ Calculate the gradient of J:
        grad = 1 / m * (X' * (sigmoid(X * theta) - y))
        grad(2 : end) += lambda / m * theta(2 : end)
        """
        m = self.x.shape[0]
        x_bias = hstack((ones((m, 1)), self.x))
        theta = atleast_2d(theta).transpose()
        grad = 1.0 / m * (dot(x_bias.transpose(), self._sigmoid(dot(x_bias, theta)) - self.y))
        grad[1 : :] += self._lambda / m * theta[1 : :]
        return grad.ravel()

    def learn(self, max_iter=300):
        """ Train theta from the dataset, return value is a 1-D array.
        """
        initial_theta = [0] * (self.x.shape[1] + 1)
        args_ = ()
        theta = fmin_bfgs(f=self._compute_cost, x0=initial_theta,
            fprime=self._compute_grad, args=args_, maxiter=max_iter)
        self._theta = atleast_2d(theta).transpose()

    def predict(self, x):
        m = x.shape[0]
        x_bias = hstack((ones((m, 1)), x))
        p = zeros((m, 1))
        prob = self._sigmoid(dot(x_bias, self._theta))
        idx = prob >= 0.5
        p[idx] = 1
        return p.ravel()

if __name__ == '__main__':
    pass
