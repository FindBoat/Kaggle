#!/usr/bin/env python
import utilities

def translate_weekday(data):
    print 'Translating weekdays...'
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
                'Friday', 'Saturday', 'Sunday']
    for i in range(len(data)):
        for j in range(len(weekdays)):
            if data[i][4] == weekdays[j]:
                data[i][4] = j + 1
    return data

def fill_NAs(data):
    print 'Filling NAs...'
    for target in range(6, len(data[0])):
        if data[0][target] == 'NA':
            for i in range(len(data)):
                if not data[i][target] == 'NA':
                    for j in range(0, i):
                        data[j][target] = data[i][target]
                    break

    for i in range(len(data)):
        for j in range(6, len(data[0])):
            if data[i][j] == 'NA':
                if i > 0 and not data[i - 1][j] == 'NA':
                    data[i][j] = data[i - 1][j]

    return data

def get_train_cv_data_by_chunk(data):
    chunk_map = utilities.get_chunk_map(data, 1)

    train_data = []
    cv_data = []
    for chunk_id in chunk_map.keys():
        num = len(chunk_map[chunk_id])
        train_num = 147
        train_data += chunk_map[chunk_id][0 : train_num]
        cv_data += chunk_map[chunk_id][train_num : :]
    return train_data, cv_data

def get_train_cv_targets(train_data, cv_data):
    return get_targets(train_data), get_targets(cv_data)

def get_targets(data):
    targets = []
    for line in data:
        n = len(line)
        targets.append(line[n - 39 : :])
    return targets

if __name__ == '__main__':
    pass
