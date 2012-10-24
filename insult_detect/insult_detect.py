#!/usr/bin/env python
import csv
from numpy import *
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
import string
import re

""" The solution is based on tf-idf text vectorization and Gaussian
Naive Bayes classification, achieving accuracy 93%. """

__train__ = './data/train.csv'
__test__ = './data/test.csv'

def read_data(path, ignore_header=True):
    csv_file_object = csv.reader(open(path, 'rb'))
    if ignore_header:
        header = csv_file_object.next()
    x = []
    for row in csv_file_object:
        x.append(row)
    return x

def feature_extract(raw_data):
    y = []
    x = []
    for row in raw_data:
        y.append(row[0])
        x.append(row[2])
    y = array(y, dtype=int32)
    return (y, x)

def comment_filter(comment):
    comment = comment.translate(string.maketrans('\n\t\r', '   '))
    comment = comment.lower()
    comment = comment.replace('\\', '')
    comment = comment.replace('\'s', '')
    comment = comment.replace('\'re', '')
    comment = re.sub(r'([^\s\w]|_)+', '', comment)
    comment = re.sub('[%s]' % string.digits, '9', comment)
    return comment

if __name__ == '__main__':
    print 'Preprocessing...'
    raw_data = read_data(__train__)
    test_data = read_data(__test__)
    y, x = feature_extract(raw_data + test_data)
    for i in range(len(x)):
        x[i] = comment_filter(x[i])

    print 'Vectorizing...'
    vectorizer = TfidfVectorizer(min_df=1, norm='l2', smooth_idf=True)
    x = vectorizer.fit_transform(x)
    x = x.toarray()

    print 'Dividing into training set and cv set...'
    num_train = len(raw_data)
    x_test = x[num_train : :, :]
    y_test = y[num_train : :]
    x = x[0 : num_train, :]
    y = y[0 : num_train]

    x_train, x_cv, y_train, y_cv = cross_validation.train_test_split(
        x, y, test_size=0.3, random_state=None)
    print 'Training set size: %d, cv set size: %d' % (
        y_train.shape[0], y_cv.shape[0])

    print 'Fitting Naive Bayes model...'
    clf = GaussianNB()
    clf.fit(x, y)

    print 'Predicting...'
    print 'Accuracy in training set: %f' % clf.score(x_train, y_train)
    print 'Accuracy in cv set: %f' % clf.score(x_cv, y_cv)

    print 'Predicting the test set...'
    p_test = clf.predict(x_test)
    open_file_object = csv.writer(open("./data/result.csv", "wb"))
    for i in range(len(test_data)):
        test_data[i][0] = p_test[i] * 1.0
        open_file_object.writerow(test_data[i])
