#!/usr/bin/env python
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
import feature_extraction
import utilities

def linear_regression_2(data):
    print 'Training with linear regression 2...'
    clf_map = {}
    positions = [1, 2, 3, 4, 5, 10, 17, 24, 48, 72]
    mae = 0.0
    num = 0
    for target in range(0, 39):
        for pos in positions:
            t = len(data[0]) - 39 + target
            key = (target, pos)
            x, y = feature_extraction.create_x_y(data)

            clf = linear_model.LinearRegression()
            clf.fit(x, y)
            clf_map[key] = clf

            p = clf.predict(x)
            mae += utilities.ae(y, p)
            num += len(y)

            print '(%s, %s) completed.' % (target, pos)
    mae /= float(num)
    print 'MAE = %s' % mae
    return clf_map

def linear_regression(x_train_all, x_cv_all, targets_train, targets_cv):
    print 'Training with linear regression...'
    clfs = regression(x_train_all, x_cv_all, targets_train, targets_cv,
                           linear_model.LinearRegression)
    return clfs

def random_forest(x_train_all, x_cv_all, targets_train, targets_cv):
    print 'Training with random forest...'
    clfs = regression(x_train_all, x_cv_all, targets_train, targets_cv,
                           m_random_forest)
    return clfs

def m_random_forest():
    return RandomForestClassifier(n_estimators=10, max_depth=None,
                                  min_samples_split=1, random_state=0)

def regression(x_train_all, x_cv_all, targets_train, targets_cv, classifier):
    clfs = []
    mae_train = 0
    mae_cv = 0
    num_train = 0
    num_cv = 0
    for i in range(len(targets_train[0])):
        x_train,  y_train, x_cv, y_cv = feature_extraction.get_x_y_by_target(
            x_train_all, x_cv_all, targets_train, targets_cv, i)

        clf = classifier()
        clf.fit(x_train, y_train)
        clfs.append(clf)

        p = clf.predict(x_cv)
        mae_cv += utilities.ae(y_cv, p)
        num_cv += len(y_cv)

        p = clf.predict(x_train)
        mae_train += utilities.ae(y_train, p)
        num_train += len(y_train)

        print 'Round %s completed.' % i

    mae_train /= float(num_train)
    mae_cv /= float(num_cv)

    print 'MAE in training set: %s' % mae_train
    print 'MAE in cv set: %s' % mae_cv
    return clfs

if __name__ == '__main__':
    pass
