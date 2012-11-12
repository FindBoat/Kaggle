#!/usr/bin/env python
from numpy import *
import utilities
import classification

def generate_features(meta_data_train, y_train, meta_data_test):
    """ Generates features for classifier. """

    # Generate maps.
    name_score_map, desc_score_map, caption_score_map, word_score_map = \
        generate_text_score_map(meta_data_train, y_train)
    geo_score_map, lat_score_map, lon_score_map = generate_geo_score_map(
        meta_data_train, y_train)
    shape_score_map, size_score_map, width_score_map, height_score_map = \
            generate_size_score_map(meta_data_train, y_train)

    # Genearte text features.
    text_features_train = generate_text_features(meta_data_train,
        name_score_map, desc_score_map, caption_score_map, word_score_map)
    text_features_test = generate_text_features(meta_data_test,
        name_score_map, desc_score_map, caption_score_map, word_score_map)

    # Generates geo features.
    geo_features_train = generate_geo_features(meta_data_train, geo_score_map,
        lat_score_map, lon_score_map)
    geo_features_test = generate_geo_features(meta_data_test, geo_score_map,
        lat_score_map, lon_score_map)

    # Generates size features
    size_features_train = generate_size_features(meta_data_train,
        shape_score_map, size_score_map, width_score_map, height_score_map)
    size_features_test = generate_size_features(meta_data_test,
        shape_score_map, size_score_map, width_score_map, height_score_map)

    # Combines all features.
    x_train = []
    for i in range(len(text_features_train)):
        x_train.append(text_features_train[i] + size_features_train[i] \
             + geo_features_train[i])

    x_test = []
    for i in range(len(text_features_test)):
        x_test.append(text_features_test[i] + size_features_test[i] \
            + geo_features_test[i])

    return (x_train, x_test)

def generate_geo_features(meta_data, geo_score_map, lat_score_map,
    lon_score_map):
    """ Generates features for geo information. """

    geo_avg_score = get_map_avg(geo_score_map)
    lat_avg_score = get_map_avg(lat_score_map)
    lon_avg_score = get_map_avg(lon_score_map)

    geo_score_features = []
    for line in meta_data:
        lat = line[0]
        lon = line[1]
        geo = (lat, lon)

        geo_score = geo_avg_score
        if geo in geo_score_map:
            geo_score = geo_score_map[geo]

        lat_score = lat_avg_score
        if lat in lat_score_map:
            lat_score = lat_score_map[lat]

        lon_score = lon_avg_score
        if lon in lon_score_map:
            lon_score = lon_score_map[lon]

        geo_score_features.append([geo_score, lat_score, lon_score])
    return geo_score_features

def generate_geo_score_map(meta_data, y):
    """ Generates score map for geo information. """

    print 'Extracting geo features...'
    geo_score_pairs = []
    lat_score_pairs = []
    lon_score_pairs = []
    for i in range(len(y)):
        lat = meta_data[i][0]
        lon = meta_data[i][1]
        geo = (lat, lon)

        geo_score_pairs.append((geo, y[i]))
        lat_score_pairs.append((lat, y[i]))
        lon_score_pairs.append((lon, y[i]))

    geo_score_map = create_key_avg_map(geo_score_pairs)
    lat_score_map = create_key_avg_map(lat_score_pairs)
    lon_score_map = create_key_avg_map(lon_score_pairs)
    return (geo_score_map, lat_score_map, lon_score_map)

def generate_size_features(meta_data, shape_score_map, size_score_map,
    width_score_map, height_score_map):
    """ Generates features for shape, size. """

    avg_shape_score = get_map_avg(shape_score_map)
    avg_size_score = get_map_avg(size_score_map)
    avg_width_score = get_map_avg(width_score_map)
    avg_height_score = get_map_avg(height_score_map)

    size_score_features = []
    for line in meta_data:
        width = line[2]
        height = line[3]
        shape = (width, height)
        size = line[4]

        shape_score = avg_shape_score
        if shape in shape_score_map:
            shape_score = shape_score_map[shape]

        size_score = avg_size_score
        if size in size_score_map:
            size_score = size_score_map[size]

        width_score = avg_width_score
        if width in width_score_map:
            width_score = width_score_map[width]

        height_score = avg_height_score
        if height in height_score_map:
            height_score = height_score_map[height]

        size_score_features.append(
            [shape_score, size_score, width_score, height_score])
    return size_score_features

def generate_size_score_map(meta_data, y):
    """ Generates score map for width, heigth, size. """

    print 'Extracting size features...'
    shape_score_pairs = []
    size_score_pairs = []
    width_score_pairs = []
    height_score_pairs = []
    for i in range(len(y)):
        width = meta_data[i][2]
        height = meta_data[i][3]
        shape = (width, height)
        size = meta_data[i][4]

        shape_score_pairs.append((shape, y[i]))
        size_score_pairs.append((size, y[i]))
        width_score_pairs.append((width, y[i]))
        height_score_pairs.append((height, y[i]))

    shape_score_map = create_key_avg_map(shape_score_pairs)
    size_score_map = create_key_avg_map(size_score_pairs)
    width_score_map = create_key_avg_map(width_score_pairs)
    height_score_map = create_key_avg_map(height_score_pairs)
    return (shape_score_map, size_score_map, width_score_map, height_score_map)

def generate_text_features(meta_data, name_score_map, desc_score_map,
    caption_score_map, word_score_map):
    """ Generates features from name, desc, caption. """

    avg_name_score = get_map_avg(name_score_map)
    avg_desc_score = get_map_avg(desc_score_map)
    avg_caption_score = get_map_avg(caption_score_map)

    text_score_features = []
    for i in range(len(meta_data)):
        name = meta_data[i][5].split(' ')
        desc = meta_data[i][6].split(' ')
        caption = meta_data[i][7].split(' ')

        name_scores = []
        for s in name:
            if s in name_score_map:
                name_scores.append(name_score_map[s])
            elif s in word_score_map:
                name_scores.append(word_score_map[s])
            else:
                name_scores.append(avg_name_score)

        desc_scores = []
        for s in desc:
            if s in desc_score_map:
                desc_scores.append(desc_score_map[s])
            elif s in word_score_map:
                desc_scores.append(word_score_map[s])
            else:
                desc_scores.append(avg_desc_score)

        caption_scores = []
        for s in caption:
            if s in caption_score_map:
                caption_scores.append(caption_score_map[s])
            elif s in word_score_map:
                caption_scores.append(word_score_map[s])
            else:
                caption_scores.append(avg_caption_score)

        # Generates features.
        name_avg_score = float(sum(name_scores)) / len(name_scores)
        desc_avg_score = float(sum(desc_scores)) / len(desc_scores)
        caption_avg_score = float(sum(caption_scores)) / len(caption_scores)

        all_scores = name_scores + desc_scores + caption_scores
        total_avg_score = float(sum(all_scores)) / len(all_scores)

        name_std = std(name_scores, name_avg_score)
        desc_std = std(desc_scores, desc_avg_score)
        caption_std = std(caption_scores, caption_avg_score)
        total_std = std(all_scores, total_avg_score)

        name_len = 0
        if name[0] != '':
            name_len = len(name)
        desc_len = 0
        if desc[0] != '':
            desc_len = len(desc)
        caption_len = 0
        if caption[0] != '':
            caption_len = len(caption)

        text_score_features.append([name_avg_score, desc_avg_score,
            caption_avg_score, total_avg_score, name_len, desc_len,
            caption_len, name_std, desc_std, caption_std, total_std])
    return text_score_features

def generate_text_score_map(meta_data, y):
    """ Generates the text score map for text features. """

    print 'Extracting text features...'
    name_y_pairs = []
    desc_y_pairs = []
    caption_y_pairs = []
    for i in range(len(y)):
        name = meta_data[i][5].split(' ')
        desc = meta_data[i][6].split(' ')
        caption = meta_data[i][7].split(' ')

        for s in name:
            name_y_pairs.append((s, y[i]))
        for s in desc:
            desc_y_pairs.append((s, y[i]))
        for s in caption:
            caption_y_pairs.append((s, y[i]))
    word_y_pairs = name_y_pairs + desc_y_pairs + caption_y_pairs

    name_score_map = create_key_avg_map(name_y_pairs)
    desc_score_map = create_key_avg_map(desc_y_pairs)
    caption_score_map = create_key_avg_map(caption_y_pairs)
    word_score_map = create_key_avg_map(word_y_pairs)
    return (name_score_map, desc_score_map, caption_score_map, word_score_map)

def std(iterable, avg):
    """ Calculate the standard deviation. """

    std = 0.0
    for n in iterable:
        std += (n - avg) ** 2
    return math.sqrt(std)

def get_map_avg(k_v_map):
    """ Calculates the average value of a map. """

    avg = 0.0
    for key in k_v_map.keys():
        avg += k_v_map[key]
    return float(avg) / len(k_v_map)

def create_key_avg_map(k_v_pairs):
    """ Creates a map which maps a key to its average value. """

    key_avg_map = {}
    for pair in k_v_pairs:
        k = pair[0]
        v = pair[1]
        if k not in key_avg_map:
            key_avg_map[k] = [v, 1]
        else:
            key_avg_map[k][0] += v
            key_avg_map[k][1] += 1

    for key in key_avg_map.keys():
        key_avg_map[key] = float(key_avg_map[key][0]) / key_avg_map[key][1]

    return key_avg_map

if __name__ == '__main__':
    pass
