#!/usr/bin/env python
import csv
import re
import operator
import Levenshtein

__train__ = './data/train.csv'
__test__ = './data/test.csv'

def read_data(path, cols,ignore_header=True):
    csv_file_object = csv.reader(open(path, 'rb'))
    if ignore_header:
        header = csv_file_object.next()
    x = []
    for row in csv_file_object:
        r = []
        for col in cols:
            r.append(row[col])
        x.append(r)
    return x

def string_normalize(s):
    res = s.lower()
    res = res.replace(' ', '')
    res = ''.join(c for c in res if c.isalnum())
    return res

def preprocess_data(raw_data, col):
    for row in raw_data:
        row[col] = string_normalize(row[col])

def best_match_key(keys, query):
    similarity = 0
    best_key = None
    for key in keys:
        sim = Levenshtein.ratio(key, query)
        if sim > similarity:
            similarity = sim
            best_key = key
    return (best_key, similarity)

def create_match(x, thresh=.85):
    match = {}
    for row in x:
        matched_key = None
        sku, query = row
        # Fuzzy matching.
        best_key, similarity = best_match_key(match.keys(), query)
        if similarity > thresh:
            matched_key = best_key
        else:
            match[query] = {sku : 1}
        if matched_key is None:
            continue
        if not match[matched_key].has_key(sku):
            match[matched_key][sku] = 1
        else:
            match[matched_key][sku] += 1

    # Sorts the dictionary.
    for key in match.keys():
        tmp_dict = match[key]
        tmp_dict = sorted(tmp_dict.iteritems(), key=operator.itemgetter(1))
        tmp_dict.reverse()
        match[key] = tmp_dict
    return match

def get_top(x):
    sku_count_dict = {}
    for row in x:
        if not sku_count_dict.has_key(row[0]):
            sku_count_dict[row[0]] = 1
        else:
            sku_count_dict[row[0]] += 1
    sorted_dict = sorted(sku_count_dict.iteritems(), key=operator.itemgetter(1))
    sorted_dict.reverse()

    res = []
    for i in range(len(sorted_dict)):
        res.append(sorted_dict[i][0])
    return res;

def predict(match, top, query, k, thresh=.7):
    res = []
    matched_key, similarity = best_match_key(match.keys(), query)
    # if similarity < 0.8:
    #     print 'matched_key = %s, query = %s, sim = %s' \
    #         % (matched_key, query, similarity)
    if similarity > thresh:
        for i in range(min(k, len(match[matched_key]))):
            res.append(match[matched_key][i][0])
    if len(res) < k:
        for i in range(len(top)):
            if top[i] not in res:
                res.append(top[i])
                if len(res) == k:
                    break
    return res

if __name__ == '__main__':
    # Reads training data.
    print 'Reading and preprocessing data...'
    x = read_data(__train__, [1, 3])
    preprocess_data(x, 1)

    # Divides into training and cv.
    l = int(len(x) * 1)
    x_cv = x[l - 10 : :]
    x = x[0 : l]

    top = get_top(x)

    # Predicts on cv.
    print 'Predicting...'
    k = 5
    # thresh_match = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
    # thresh_predict = [0, 0.2, 0.4, 0.6, 0.7, 0.8]
    thresh_match = [0.75]
    thresh_predict = [0]
    best_match = None
    thresh1 = 0
    thresh2 = 0
    accuracy = -1
    for t1 in thresh_match:
        for t2 in thresh_predict:
            match = create_match(x, t1)
            p_cv = []
            for row in x_cv:
                q = row[1]
                p_cv.append(predict(match, top, q, k, t2))
            correct = 0.
            for i in range(len(x_cv)):
                if x_cv[i][0] in p_cv[i]:
                    correct += 1.
            ac = correct / len(x_cv)
            print 't1 = %f, t2 = %f, accuracy = %f' % (t1, t2, ac)
            if ac > accuracy:
                accuracy = ac
                thresh1 = t1
                thresh2 = t2
                best_match = match
    print 'thresh1 = %f, thresh2 = %f, accuracy = %f' \
        % (thresh1, thresh2, accuracy)

    # Reads test set.
    x_test = read_data(__test__, [2])
    preprocess_data(x_test, 0)

    # Predicts.
    res = []
    k = 5
    for row in x_test:
        q = row[0]
        res.append(predict(best_match, top, q, k, thresh2))

    open_file_object = csv.writer(open("./data/result.csv", "wb"))
    open_file_object.writerow(['sku'])
    for p in res:
        open_file_object.writerow([' '.join(p)])

