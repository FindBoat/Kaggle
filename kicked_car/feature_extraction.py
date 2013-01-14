#!/usr/bin/env python

from sklearn.tree import DecisionTreeClassifier
from collections import defaultdict
import math

_NUM_FEATURE_INDICES = [3, 12, 16, 17, 18, 19, 20, 21, 22, 23, 29, 31]
_CATE_FEATURE_INDICES = [0, 1, 2, 4, 6, 8, 9, 10, 11, 13, 14, 15, 24, 26,
                        27, 28, 30, 32]

def get_loglikelihood_ratio(x, y, idx, cate_range):
    pos_num = 0
    neg_num = 0
    pos_map = defaultdict(int)
    neg_map = defaultdict(int)
    for i in range(len(x)):
        cate = x[i][idx]
        if y[i] == 1:
            pos_num += 1
            pos_map[cate] += 1
        else:
            neg_num += 1
            neg_map[cate] += 1

    ratio_map = defaultdict(lambda: 0)
    for cate in range(cate_range + 1):
        p_pos = -100
        if cate in pos_map:
            p_pos = math.log10(pos_map[cate] / float(pos_num))
        p_neg = -100
        if cate in neg_map:
            p_neg = math.log10(neg_map[cate] / float(neg_num))
        ratio_map[cate] = p_pos - p_neg
    return ratio_map

def get_feature(x, range_map, ratio_map):
    x_new = []
    for line in x:
        # Numerical features.
        features = [line[idx] for idx in _NUM_FEATURE_INDICES]
        # Cur - avg.
        features.append(line[17] - line[16])
        features.append(line[19] - line[18])
        features.append(line[21] - line[20])
        features.append(line[23] - line[22])

        # Diff cur.
        features.append(line[19] - line[17])
        features.append(line[21] - line[19])
        features.append(line[23] - line[21])
        features.append(line[21] - line[17])
        features.append(line[23] - line[17])
        features.append(line[23] - line[19])

        # Diff avg.
        features.append(line[18] - line[16])
        features.append(line[20] - line[18])
        features.append(line[22] - line[20])
        features.append(line[22] - line[18])

        # Categorical features.
        for idx in _CATE_FEATURE_INDICES:
            for i in range(range_map[idx] + 1):
                if i == line[idx]:
                    features.append(1)
                else:
                    features.append(0)

        # Log likelihood ratio
        for idx in _CATE_FEATURE_INDICES:
            cate = line[idx]
            cur_ratio_map = ratio_map[idx]
            features.append(cur_ratio_map[cate])

        x_new.append(features)
    return x_new

def create_feature(x, y, x_test):
    range_map = {}
    ratio_map = {}
    for idx in _CATE_FEATURE_INDICES:
        range_map[idx] = max(x, key=lambda s: s[idx])[idx]
        ratio_map[idx] = get_loglikelihood_ratio(x, y, idx, range_map[idx])

    x_new = get_feature(x, range_map, ratio_map)
    x_test_new = get_feature(x_test, range_map, ratio_map)
    return (x_new, x_test_new)

def get_best_k_feature_indices(x, y, k):
    print 'Getting best k features...'
    clf = DecisionTreeClassifier(random_state=0, compute_importances=True)
    clf.fit(x, y)
    importance_pairs = [(i, clf.feature_importances_[i])
                        for i in range(len(clf.feature_importances_))]
    importance_pairs = sorted(importance_pairs, key=lambda s: s[1])
    return [importance_pairs[i][0] for i in range(k)]

def get_best_k_features(x, indices):
    x_important = []
    for line in x:
        features = [line[idx] for idx in indices]
        x_important.append(features)
    return x_important

if __name__ == '__main__':
    pass
