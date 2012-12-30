#!/usr/bin/env python
import preprocess
import utilities

def create_x_y(data, target, pos):
    span = 24
    x = []
    y = []
    for i in range(len(data)):
        if i % 192 < pos + span:
            continue
        chunk_id = data[i][1]
        hour = data[i][5]
        t = len(data[0]) - 39 + target

        features = []
        prev_hour = 0
        for j in range(i - pos - span, i - pos):
            features.append(float(data[j][t]))
            if data[j][5] == hour:
                prev_hour = float(data[j][t])

        features.append(prev_hour)

        # Binary hour features.
        for h in range(24):
            if h == int(hour):
                features.append(1)
            else:
                features.append(0)

        # Binary month features.
        month = int(data[i][3])
        for m in range(1, 13):
            if m == month:
                features.append(1)
            else:
                features.append(0)

        # Weather features.
        for k in range(6, 56):
            features.append(float(data[i - pos][k]))
        for k in range(6, 56):
            features.append(float(data[i - pos - 1][k]))

        x.append(features)

        y.append(float(data[i][t]))

    return x, y

def get_features(chunk_id, weekday, hour, chunk_avg, hour_avg_by_chunk,
                 weekday_avg_by_chunk, hour_avg, weekday_avg):
    avg = [0.0] * 39
    for chunk_id in chunk_avg.keys():
        for i in range(len(avg)):
            avg[i] += chunk_avg[chunk_id][i]

    for i in range(len(avg)):
        avg[i] /= float(len(chunk_avg))

    tmp = []
    if chunk_id in chunk_avg:
        tmp.append(chunk_avg[chunk_id])
    else:
        tmp.append(avg)
    # if weekday in weekday_avg_by_chunk[chunk_id]:
    #     tmp.append(weekday_avg_by_chunk[chunk_id][weekday])
    # else:
    #     tmp.append(weekday_avg[weekday])
    if chunk_id in chunk_avg and hour in hour_avg_by_chunk[chunk_id]:
        tmp.append(hour_avg_by_chunk[chunk_id][hour])
    else:
        tmp.append(hour_avg[hour])
    return tmp

def get_avgs(data, chunk_avg, hour_avg_by_chunk, weekday_avg_by_chunk,
             hour_avg, weekday_avg):
    res = []
    for line in data:
        chunk_id = line[1]
        weekday = line[4]
        hour = line[5]

        tmp = get_features(chunk_id, weekday, hour, chunk_avg,
                           hour_avg_by_chunk, weekday_avg_by_chunk,
                           hour_avg, weekday_avg)
        res.append(tmp)
    return res

def get_avg_maps(train_data):
    chunk_avg = utilities.get_chunk_avg(train_data)
    hour_avg = utilities.get_hour_avg(train_data)
    hour_avg_by_chunk = utilities.get_hour_avg_by_chunk(train_data)
    weekday_avg = utilities.get_weekday_avg(train_data)
    weekday_avg_by_chunk = utilities.get_weekday_avg_by_chunk(train_data)

    return (chunk_avg, hour_avg_by_chunk, weekday_avg_by_chunk,
            hour_avg, weekday_avg)

def get_x_by_avg(train_data, cv_data, chunk_avg, hour_avg_by_chunk,
                 weekday_avg_by_chunk, hour_avg, weekday_avg):
    x_train = get_avgs(train_data, chunk_avg, hour_avg_by_chunk,
                       weekday_avg_by_chunk, hour_avg, weekday_avg)
    x_cv = get_avgs(cv_data, chunk_avg, hour_avg_by_chunk,
                       weekday_avg_by_chunk, hour_avg, weekday_avg)
    return x_train, x_cv

def get_x_y_by_target(x_train_all, x_cv_all, targets_train, targets_cv, index):
    x_train = []
    y_train = []
    for i in range(len(targets_train)):
        if not targets_train[i][index] == 'NA':
            tmp = []
            for features in x_train_all[i]:
                tmp.append(features[index])
            x_train.append(tmp)
            y_train.append(float(targets_train[i][index]))

    x_cv = []
    y_cv = []
    for i in range(len(targets_cv)):
        if not targets_cv[i][index] == 'NA':
            tmp = []
            for features in x_cv_all[i]:
                tmp.append(features[index])
            x_cv.append(tmp)
            y_cv.append(float(targets_cv[i][index]))

    return x_train, y_train, x_cv, y_cv

if __name__ == '__main__':
    pass
