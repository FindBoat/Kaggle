#!/usr/bin/env python
from random import shuffle

def extract_year_month(x):
    idx = 0
    for i in range(len(x)):
        pos = x[i][idx].rfind('/', 0, len(x[i][idx]))
        pos0 = x[i][idx].find('/', 0, len(x[i][idx]))
        x[i].append(x[i][idx][0 : pos0])
        x[i][idx] = x[i][idx][pos + 1 : :]

def create_category_map(x, idx):
    category_map = {}
    cur = 0
    for line in x:
        cate = line[idx]
        if not cate in category_map:
            category_map[cate] = cur
            cur += 1
    return category_map

def convert_category_to_int(x, idx, category_map):
    cur = max(category_map.values()) + 1
    for i in range(len(x)):
        cate = x[i][idx]
        if cate in category_map:
            cate_num = category_map[cate]
        else:
            category_map[cate] = cur
            cur += 1
        x[i][idx] = cate_num
    return x

def convert_categories(x, x_test):
    cate_feature_indices = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15,
                            24, 25, 26, 27, 28, 30, 32]
    for idx in cate_feature_indices:
        cate_map = create_category_map(x, idx)
        x = convert_category_to_int(x, idx, cate_map)
        x_test = convert_category_to_int(x_test, idx, cate_map)
    return (x, x_test)

def get_single_numerical_median(x, idx):
    all_values = [float(line[idx]) for line in x
                  if not line[idx] == '' and not line[idx] == 'NULL']
    all_values.sort()
    return all_values[len(all_values) / 2]

def fill_missing_numerical_feature(x, idx, median):
    for i in range(len(x)):
        if x[i][idx] == '' or x[i][idx] == 'NULL':
            x[i][idx] = median
        else:
            x[i][idx] = float(x[i][idx])
    return x

def fill_numerical_features(x, x_test):
    num_feature_indices = [3, 12, 16, 17, 18, 19, 20, 21, 22, 23, 29, 31]
    for idx in num_feature_indices:
        median = get_single_numerical_median(x, idx)
        x = fill_missing_numerical_feature(x, idx, median)
        x_test = fill_missing_numerical_feature(x_test, idx, median)
    return (x, x_test)

def preprocess_features(x, x_test):
    extract_year_month(x)
    extract_year_month(x_test)
    x, x_test = fill_numerical_features(x, x_test)
    x, x_test= convert_categories(x, x_test)
    return (x, x_test)

def down_sample(x, y, ratio):
    print 'Down sampling...'
    pos_indices = [i for i in range(len(y)) if y[i] == 1]
    neg_indices = [i for i in range(len(y)) if y[i] == 0]

    neg_num = min(int(len(pos_indices) * ratio), len(neg_indices))
    shuffle(neg_indices)
    sample_indices = pos_indices + neg_indices[0 : neg_num]
    shuffle(sample_indices)

    # Down sampling.
    x_ds = [x[idx] for idx in sample_indices]
    y_ds = [y[idx] for idx in sample_indices]
    return (x_ds, y_ds)

def prepare_data(data):
    x = [line[2 : :] for line in data]
    y = [int(line[1]) for line in data]
    return (x, y)

def prepare_test_data(data):
    x = [line[1 : :] for line in data]
    refid = [line[0] for line in data]
    return (refid, x)

if __name__ == '__main__':
    pass
