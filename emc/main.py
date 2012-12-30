#!/usr/bin/env python
import time
import utilities
import preprocess
import feature_extraction
import regression

def time_series(training_file, submission_file, output_file):
    data = utilities.read_file(training_file, True)
    first_line = data[0]
    data = data[1 : :]
    data = preprocess.fill_NAs(data)

    (chunk_avg, hour_avg_by_chunk, weekday_avg_by_chunk,
     hour_avg, weekday_avg) = feature_extraction.get_avg_maps(data)

    clf_map = regression.linear_regression_2(data)

    print 'Filling submission file...'
    chunk_map = utilities.get_chunk_map(data, 1)
    sub_data = utilities.read_file(submission_file, True)

    positions = [1, 2, 3, 4, 5, 10, 17, 24, 48, 72]
    for i in range(1, len(sub_data)):
        chunk_id = sub_data[i][1]
        hour = sub_data[i][3]
        pos = positions[(i - 1) % 10]
        for j in range(5, len(sub_data[i])):
            target = j - 5
            if sub_data[i][j] == '0':
                if not chunk_id in chunk_map:
                    sub_data[i][j] = hour_avg[hour][target]
                else:
                    data_in_chunk = chunk_map[chunk_id]
                    start = len(data_in_chunk) - 24
                    t = len(data_in_chunk[0]) - 39 + target
                    features = []
                    prev_hour = 0
                    for k in range(start, len(data_in_chunk)):
                        features.append(float(data_in_chunk[k][t]))
                        if data_in_chunk[k][5] == hour:
                            prev_hour = float(data_in_chunk[k][t])

                    features.append(prev_hour)

                    # Binary hour features.
                    for h in range(24):
                        if h == int(hour):
                            features.append(1)
                        else:
                            features.append(0)

                    # Binary month features.
                    month = int(sub_data[i][4])
                    for m in range(1, 13):
                        if m == month:
                            features.append(1)
                        else:
                            features.append(0)

                    # Weather features.
                    tmp_length = len(data_in_chunk)
                    for k in range(6, 56):
                        features.append(float(data_in_chunk[tmp_length - 1][k]))
                    for k in range(6, 56):
                        features.append(float(data_in_chunk[tmp_length - 2][k]))

                    sub_data[i][j] = \
                        clf_map[(target, pos)].predict([features])[0]

    utilities.write_file(output_file, sub_data)

def avg(training_file, submission_file, output_file):
    data = utilities.read_file(training_file)

    train_data, cv_data = preprocess.get_train_cv_data_by_chunk(data)
    targets_train, targets_cv = preprocess.get_train_cv_targets(
        train_data, cv_data)

    (chunk_avg, hour_avg_by_chunk, weekday_avg_by_chunk,
     hour_avg, weekday_avg) = feature_extraction.get_avg_maps(train_data)

    x_train_all, x_cv_all = feature_extraction.get_x_by_avg(
            train_data, cv_data, chunk_avg, hour_avg_by_chunk,
             weekday_avg_by_chunk, hour_avg, weekday_avg)

    clfs = regression.linear_regression(
        x_train_all, x_cv_all, targets_train, targets_cv)
    clfs = regression.random_forest(
        x_train_all, x_cv_all, targets_train, targets_cv)

    print 'Filling submission file...'
    sub_data = utilities.read_file(submission_file, True)
    for i in range(1, len(sub_data)):
        chunk_id = sub_data[i][1]
        hour = sub_data[i][3]
        weekday = ''
        all_features = feature_extraction.get_features(
            chunk_id, weekday, hour, chunk_avg, hour_avg_by_chunk,
            weekday_avg_by_chunk, hour_avg, weekday_avg)

        for j in range(5, len(sub_data[i])):
            if sub_data[i][j] == '0':
                feature = []
                for f in all_features:
                    feature.append(f[j - 5])
                sub_data[i][j] = clfs[j - 5].predict([feature])[0]

    utilities.write_file(output_file, sub_data)

def baseline(training_file, submission_file, output_file):
    data = utilities.read_file(training_file)
    sub_data = utilities.read_file(submission_file, True)

    print 'Calculating hour averages...'
    hour_avg_by_chunk = utilities.get_hour_avg_by_chunk(data)
    hour_avg = utilities.get_hour_avg(data)

    print 'Filling submission file...'
    for i in range(1, len(sub_data)):
        chunk_id = sub_data[i][1]
        hour = sub_data[i][3]
        for j in range(5, len(sub_data[i])):
            if sub_data[i][j] == '0':
                if chunk_id in hour_avg_by_chunk:
                    sub_data[i][j] = hour_avg_by_chunk[chunk_id][hour][j - 5]
                else:
                    sub_data[i][j] = hour_avg[hour][j - 5]

    utilities.write_file(output_file, sub_data)

if __name__ == '__main__':
    start_time = time.time()
    time_series('./data/TrainingData.csv',
             './data/SubmissionZerosExceptNAs.csv',
             './data/result.csv')
    print (time.time() - start_time) / 60.0, 'minutes'
