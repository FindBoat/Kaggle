#!/usr/bin/env python

from numpy import *
from sklearn import cross_validation
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier

def prepare_data(x, y, size=0.3, state=0):
    """ Divides data into training set and cross validation set. """

    x_train, x_cv, y_train, y_cv = cross_validation.train_test_split(
        x, y, test_size=size, random_state=state)

    return (x_train, y_train, x_cv, y_cv)

def knn(x_train, y_train, x_cv, y_cv, k=3):
    """ Using KNN to classify the data. """

    print 'Training with KNN...'
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(x_train, y_train)

    print 'Accuracy in training set: %f' % clf.score(x_train, y_train)
    print 'Accuracy in cv set: %f' % clf.score(x_cv, y_cv)
    return clf

def bernoulli_naive_bayes(x_train, y_train, x_cv, y_cv):
    """ Using Naive Bayes to classify the data. """

    print 'Training with NB...'
    clf = BernoulliNB()
    clf.fit(x_train, y_train)

    print 'Accuracy in training set: %f' % clf.score(x_train, y_train)
    print 'Accuracy in cv set: %f' % clf.score(x_cv, y_cv)
    return clf

def naive_bayes(x_train, y_train, x_cv, y_cv):
    """ Using Naive Bayes to classify the data. """

    print 'Training with NB...'
    clf = MultinomialNB()
    clf.fit(x_train, y_train)

    print 'Accuracy in training set: %f' % clf.score(x_train, y_train)
    print 'Accuracy in cv set: %f' % clf.score(x_cv, y_cv)
    return clf

def random_forest(x_train, y_train, x_cv, y_cv):
    """ Using Random Forest to classify the data. """

    print 'Training with RF...'
    clf = RandomForestClassifier(n_estimators = 2000, max_features=2)
    clf.fit(x_train, y_train)

    print 'Predicting...'
    print 'Accuracy in training set: %f' % clf.score(x_train, y_train)
    if y_cv != None:
        print 'Accuracy in cv set: %f' % clf.score(x_cv, y_cv)
    return clf

def logistic_regression(x_train, y_train, x_cv, y_cv):
    """ Using Logistic Regression to classify the data. """

    print 'Training with LR...'
    clf = linear_model.LogisticRegression(penalty='l2', C=.03)
    clf.fit(x_train, y_train)

    print 'Accuracy in training set: %f' % clf.score(x_train, y_train)
    if y_cv != None:
        print 'Accuracy in cv set: %f' % clf.score(x_cv, y_cv)
    return clf

def get_prob(clf, x):
    """ Gets the probability of being good. """

    prob = array(clf.predict_proba(x))
    return prob[:, 1]

if __name__ == '__main__':
    pass
