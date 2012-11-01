"""
Responsible for extracting features, classification and ranking.
"""

#!/usr/bin/env python
import utilities
from numpy import *
from sklearn import cross_validation
from sklearn import linear_model
from sklearn.metrics import precision_recall_fscore_support

def get_features(follow, followed, n1, n2):
    """ Creates features for a given pair of nodes. """

    # Level 1 features.
    does_follow = 0
    if n1 in follow[n2]:
        does_follow = 1

    # Level 2 features.
    followees_follow = set.intersection(follow[n1], followed[n2])
    percent_followees_follow = 0.0
    if len(follow[n1]) > 0:
        percent_followees_follow = 1.0 * len(followees_follow) / len(follow[n1])

    followees_followed = set.intersection(follow[n1], follow[n2])
    percent_followees_followed = 0.0
    if len(follow[n1]) > 0:
        percent_followees_followed = 1.0 * len(followees_followed) \
            / len(follow[n1])

    followers_follow = set.intersection(followed[n1], followed[n2])
    percent_followers_follow = 0.0
    if len(followed[n1]) > 0:
        percent_followers_follow = 1.0 * len(followers_follow) \
            / len(followed[n1])

    followers_followed = set.intersection(followed[n1], follow[n2])
    percent_followers_followed = 0.0
    if len(followed[n1]) > 0:
        percent_followers_followed = 1.0 * len(followers_followed) \
            / len(followed[n1])

    return [does_follow, percent_followees_follow, percent_followees_followed,
            percent_followers_follow, percent_followers_followed]

def rank_candidates(follow, followed, clf, node, candidates):
    """ Ranks the candidates based on the chance they will be followed. """

    if not candidates:
        return []

    # Generates feature matrix.
    candidates = list(candidates)
    x_candidates = []
    for candidate in candidates:
        features = get_features(follow, followed, node, candidate)
        x_candidates.append(features)

    # Uses classifier to estimate probability.
    candidate_score = {}
    prob = clf.predict_proba(x_candidates)
    for i in range(len(candidates)):
        candidate_score[candidates[i]] = prob[i][1]

    # Ranks candidates based on the score
    return  sorted(candidate_score, key=candidate_score.__getitem__,
        reverse=True)

def get_data(data_file, test_file):
    """ Produces training set, cross validation set and test set. """

    raw_data = utilities.read_file(data_file, True)
    test_data = utilities.read_file(test_file, True)
    x = array(raw_data, float64)
    y = x[:, 0]
    x = x[:, 1 : :]
    x_train, x_cv, y_train, y_cv = cross_validation.train_test_split(
        x, y, test_size=0.3, random_state=None)
    x = array(test_data, float64)
    y_test = x[:, 0]
    x_test = x[:, 1 : :]

    return (x_train, y_train, x_cv, y_cv, x_test, y_test)

def train(data_file, test_file):
    """ Uses random forest to train the model. """

    x_train, y_train, x_cv, y_cv, x_test, y_test = get_data(data_file,
        test_file)

    clf = linear_model.LogisticRegression(penalty='l1', C=1)
    clf.fit(x_train, y_train)
    print clf.coef_

    print 'Accuracy in training set: %f'% clf.score(x_train, y_train)
    print 'Accuracy in cv: %f' %  clf.score(x_cv, y_cv)
    print 'Accuracy in test: %f' %  clf.score(x_test, y_test)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, clf.predict(x_test))
    print precision, recall, f1

    return clf

if __name__ == '__main__':
    train('./data/data.csv',
          './data/data_test.csv')

