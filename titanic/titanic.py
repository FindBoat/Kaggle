#!/usr/bin/env python
import csv
from numpy import *
from logistic_regression import LogisticRegression

def map_feature(x):
    """ Add polynomial features to x in order to reduce high bias.
    """
    m, n = x.shape
    out = x

    # Add quodratic features.
    for i in range(n):
        for j in range(i, n):
            out = hstack((out, x[:, i].reshape(m, 1) * x[:, j].reshape(m, 1)))

    # Add cubic features.
    for i in range(n):
        for j in range(i, n):
            for k in range(j, n):
                out = hstack(
                    (out, x[:, i].reshape(m, 1) * x[:, j].reshape(m, 1) * x[:, k].reshape(m, 1)))
    return out

def scale_data(x):
    """ Scale data with zero mean and unit variance.
    """
    mu = x.mean(axis=0)
    sigma = x.std(axis=0)
    x = (x - mu) / sigma
    return (x, mu, sigma)

def read_data():
    # Data in the file has been preprocessed by eliminating rows with missing values.
    csv_file_object = csv.reader(open('./data/data.csv', 'rb')) 
    header = csv_file_object.next()
    x = []
    for row in csv_file_object:
        x.append(row)
    return array(x, dtype=float64)

if __name__ == '__main__':
    x = read_data()

    # Generates training set and cross validation set.
    y = x[:, 0]
    x = x[:, 1 : :]
    x = map_feature(x)
    num = int(x.shape[0] * .7)
    x_cv = x[num : :, :]
    y_cv = y[num : :]
    x = x[0 : num, :]
    y = y[0 : num]

    # Feature scaling.
    x, mu, sigma = scale_data(x)
    x_cv = (x_cv - mu) / sigma

    # Use cross validation set to find the best lambda for regularization.
    C_candidates = [0, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    lambda_ = 0
    best_accuracy = 0
    for C in C_candidates:
        clf = LogisticRegression(x, y, C)
        clf.learn()
        p_cv = clf.predict(x_cv)
        accuracy = (p_cv == y_cv).mean()
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            lambda_ = C
    print 'Best regularization parameter lambda: %f' % lambda_

    clf = LogisticRegression(x, y, lambda_)
    clf.learn()
    p = clf.predict(x)
    p_cv = clf.predict(x_cv)
    print 'Accuracy in training set: %f'% (p == y).mean()
    print 'Accuracy in cv: %f' %  (p_cv == y_cv).mean()
