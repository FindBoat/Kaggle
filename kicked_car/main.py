#!/usr/bin/env python

import time
import utilities
import preprocess
import classification
import feature_extraction
from sklearn import cross_validation

def main(training_file, test_file, submission_file, ratio):
    data = utilities.read_file(training_file)
    test_data = utilities.read_file(test_file)

    print 'Preparing data...'
    x, y = preprocess.prepare_data(data)
    refid, x_test = preprocess.prepare_test_data(test_data)
    x, x_test = preprocess.preprocess_features(x, x_test)

    print 'Feature extracting...'
    x, x_test = feature_extraction.create_feature(x, y, x_test)

    indices = feature_extraction.get_best_k_feature_indices(x, y, 300)
    x = feature_extraction.get_best_k_features(x, indices)
    x_test = feature_extraction.get_best_k_features(x_test, indices)
    print 'Get %s features.' % len(x[0])

    x_train, x_cv, y_train, y_cv = cross_validation.train_test_split(
        x, y, test_size=.3, random_state=0)
    x_train, y_train = preprocess.down_sample(x_train, y_train, ratio)

    clf = classification.random_forest(x_train, y_train, x_cv, y_cv)

    print 'Predicting...'
    predict = clf.predict_proba(x_test)
    utilities.write_submission_file(submission_file, refid, predict)

if __name__ == '__main__':
    start_time = time.time()

    main('./data/training.csv',
       './data/test.csv',
       './data/res.csv',
       1.5)

    print (time.time() - start_time) / 60.0, 'minutes'
