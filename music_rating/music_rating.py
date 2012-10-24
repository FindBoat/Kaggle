#!/usr/bin/env python
import csv
from numpy import *
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn import linear_model

__train__ = './data/train.csv'
__test__ = './data/test.csv'
__users__ = './data/users.csv'
__words__ = './data/words.csv'

__music_clf_map__ = {}

def read_data(path, ignore_header=True,  max_line=-1):
    """ Reads data from file. """
    csv_file_object = csv.reader(open(path, 'rb'))
    if ignore_header:
        header = csv_file_object.next()
    x = []
    for row in csv_file_object:
        if max_line >= 0 and len(row) >= max_line:
            break
        x.append(row)
    return x

def generate_user_features():
    """ Generate user features from users.csv.
    Features include sex, age, questions."""
    profiles = read_data(__users__)
    indices_features = [0, 1, 2] + range(8, 27)
    user_map = {}
    for row in profiles:
        features = []
        for index in indices_features:
            features.append(row[index])
        if features[1] == 'Male':
            features[1] = 0
        else:
            features[1] = 1
        user_map[features[0]] = features[1 : :]
    return user_map

def get_mean(user_map, num_features):
    mean = [0.0] * num_features
    count = [0] * num_features
    for key in user_map.keys():
        features = user_map[key]
        for i in range(num_features):
            if features[i] != '':
                mean[i] += float(features[i])
                count[i] += 1
    for i in range(num_features):
        mean[i] /= count[i]
    return mean

def get_std(user_map, num_features, mean):
    std = [0.0] * num_features
    count = [0] * num_features
    for key in user_map.keys():
        features = user_map[key]
        for i in range(num_features):
            if features[i] != '':
                std[i] += (float(features[i]) - mean[i]) ** 2
                count[i] += 1
    for i in range(num_features):
        std[i] = math.sqrt(std[i] / count[i])
    return std

def preprocess_feature(user_map):
    """ Fills empty features with averages and scales the data."""
    num_features = 21
    mean = get_mean(user_map, num_features)
    std = get_std(user_map, num_features, mean)
    # Scaling.
    for key in user_map.keys():
        features = user_map[key]
        for i in range(len(features)):
            if features[i] == '':
                features[i] = 0.0
            else:
                features[i] = (float(features[i]) - mean[i]) / std[i]

def extract_rating(data, artist):
    """ Extracts all the data includes rating, track, user etc. given
    an artist id. """
    ratings = []
    for row in data:
        if row[0] == artist:
            ratings.append(row)
    return ratings

def generate_train_set(user_map, ratings, artist_user_pref):
    """ Generates training set based on all ratings of a particular artist,
    features combine both user profile and features from word.csv. """
    x = []
    y = []
    cnt = 0
    for row in ratings:
        if user_map.has_key(row[2]):
            artist_user = (row[0], row[2])
            if artist_user_pref.has_key(artist_user):
                y.append(row[3])
                x.append(user_map[row[2]] + artist_user_pref[artist_user])
            else:
                cnt += 1
    print cnt
    x = array(x, float64)
    y = array(y, float64)
    return (x, y)

def rmse(real_value, predict_value):
    """ Calculating RMSE error. """
    rmse = 0.0
    for i in range(real_value.shape[0]):
        rmse += (real_value[i] - predict_value[i]) ** 2
    rmse = math.sqrt(rmse / real_value.shape[0])
    return rmse

def generate_music_clf_map(data, artist_user_pref):
    """ Generates classifiers for each artist. """
    for row in data:
        artist = row[0]
        if __music_clf_map__.has_key(artist):
            continue
        ratings = extract_rating(data, artist)
        x, y = generate_train_set(user_map, ratings, artist_user_pref)
        clf = linear_model.Lasso(alpha=.5)
        clf.fit(x, y)
        __music_clf_map__[artist] = clf
        print 'RMSE for %s: %f' % (artist, rmse(y, clf.predict(x)))

def generate_artist_user_pref():
    """ Generates features for each (artist, user) pair from word.csv. """
    words = read_data(__words__)
    artist_user_pref = {}
    for row in words:
        artist_user = (row[0], row[1])
        pref = row[4 : :]
        for i in range(len(pref)):
            if pref[i] == '':
                pref[i] = 0.0
            else:
                pref[i] = float(pref[i])
        if len(pref) == 82:
            pref.append(0)
        artist_user_pref[artist_user] = pref
    return artist_user_pref

def generate_artist_mean(data):
    """ Calculate average rating for each artist. """
    artist_mean = {}
    artist_rate = {}
    for row in data:
        artist = row[0]
        rate = row[3]
        if artist_rate.has_key(artist):
            artist_rate[artist].append(float(rate))
        else:
            artist_rate[artist] = [float(rate)]
    for key in artist_rate.keys():
        artist_mean[key] = sum(artist_rate[key]) / len(artist_rate[key])
    return artist_mean

if __name__ == '__main__':
    print 'Generating user features...'
    user_map = generate_user_features()
    preprocess_feature(user_map)
    data = read_data(__train__)
    artist_mean = generate_artist_mean(data)
    artist_user_pref = generate_artist_user_pref()

    print 'Generating classifiers for each artist...'
    generate_music_clf_map(data, artist_user_pref)
    test_data = read_data(__test__)
    p_test = []
    for row in test_data:
        miss = False
        feature = None
        artist = row[0]
        uid = row[2]
        clf = __music_clf_map__[artist]
        if user_map.has_key(uid):
            feature = list(user_map[uid])
            if artist_user_pref.has_key((artist, uid)):
                feature += artist_user_pref[(artist, uid)]
            else:
                miss = True
        else:
            miss = True
        if not miss:
            p_test.append(clf.predict(feature))
        else:
            # Uses average ratings when user cannot be found.
            p_test.append(artist_mean[artist])

    open_file_object = csv.writer(open("./data/result.csv", "wb"))
    for p in p_test:
        open_file_object.writerow([p])
