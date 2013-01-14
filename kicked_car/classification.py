#!/usr/bin/env python

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support

def random_forest(x_train, y_train, x_cv, y_cv):
    """ Using Random Forest to classify the data. """

    print 'Training with RF...'
    clf = RandomForestClassifier(n_estimators = 10)
    clf.fit(x_train, y_train)

    print 'Predicting...'
    print 'Accuracy in training set: %f' % clf.score(x_train, y_train)
    if y_cv != None:
        print 'Accuracy in cv set: %f' % clf.score(x_cv, y_cv)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_cv, clf.predict(x_cv))
        print precision, recall, f1

    return clf

if __name__ == '__main__':
    pass
