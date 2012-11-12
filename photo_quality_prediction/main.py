#!/usr/bin/env python
import time
import utilities
import classification
import feature_selection

def all_feature_classify(training_file, num):
    """ Classifier using all features. """

    y, meta_data = utilities.read_training_file(training_file)
    y, meta_data = utilities.sample(y, meta_data, num)

    meta_data_train, y_train, meta_data_cv, y_cv = \
        classification.prepare_data(meta_data, y)

    x_train, x_cv = feature_selection.generate_features(meta_data_train,
        y_train, meta_data_cv)

    clf = classification.random_forest(x_train, y_train, x_cv, y_cv)
    print utilities.binomial_deviance(y_train,
        classification.get_prob(clf, x_train))
    print utilities.binomial_deviance(y_cv, classification.get_prob(clf, x_cv))

def spring_brother(training_file, test_file, submission_file):
    """ Running on the test file. """

    y, meta_data = utilities.read_training_file(training_file)
    ids, meta_data_test = utilities.read_test_file(test_file)

    x_train, x_test = feature_selection.generate_features(meta_data,
        y, meta_data_test)

    clf = classification.random_forest(x_train, y, None, None)

    p = classification.get_prob(clf, x_test)
    utilities.write_submission_file(submission_file, ids, p)

if __name__ == '__main__':
    start_time = time.time()

    spring_brother('./data/training.csv',
        './data/test.csv',
        './data/result.csv')

#    all_feature_classify('./data/training.csv', 40000)

    print (time.time() - start_time) / 60.0, 'minutes'
