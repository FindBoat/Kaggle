#!/usr/bin/env python
import csv

def read_file(file_name, header=False):
    print 'Reading file...'
    f = open(file_name)
    reader = csv.reader(f)
    res = []
    if not header:
        reader.next()
    for line in reader:
        res.append(line)

    f.close()
    return res

def write_file(file_name, data):
    print 'Writing submission file...'
    f = open(file_name, 'w')
    writer = csv.writer(f)
    for line in data:
        writer.writerow(line)
    f.close()

def get_site_map(first_line):
    res = []
    site_map = {}
    index = 0
    start = len(first_line) - 39
    for i in range(start, len(first_line)):
        site = first_line[i]
        idx = site.rfind('_', 0, len(site))
        site_num = int(site[idx + 1 : :])
        if site_num not in site_map:
            site_map[site_num] = index
            index += 1

        res.append(site_map[site_num])
    return res

def get_chunk_map(data, index):
    chunk_map = {}
    for line in data:
        key = line[index]
        if key not in chunk_map:
            chunk_map[key] = []
        chunk_map[key].append(line)
    return chunk_map

def get_avg_by_index(data, index):
    avg = {}
    num = {}
    for line in data:
        key = line[index]
        if key not in avg:
            avg[key] = [0.0] * 39
            num[key] = [0] * 39
        for i in range(56, len(line)):
            if not line[i] == 'NA':
                num[key][i - 56] += 1
                avg[key][i - 56] += float(line[i])

    for key in avg.keys():
        for i in range(len(avg[key])):
            if num[key][i] > 0:
                avg[key][i] /= float(num[key][i])
    return avg

def get_chunk_avg(data):
    return get_avg_by_index(data, 1)

def get_hour_avg(data):
    return get_avg_by_index(data, 5)

def get_weekday_avg(data):
    return get_avg_by_index(data, 4)

def get_hour_avg_by_chunk(data):
    chunk_map = get_chunk_map(data, 1)

    hour_avg_by_chunk = {}
    for chunk_id in chunk_map.keys():
        hour_avg_by_chunk[chunk_id] = get_hour_avg(chunk_map[chunk_id])
    return hour_avg_by_chunk

def get_weekday_avg_by_chunk(data):
    chunk_map = get_chunk_map(data, 1)

    weekday_avg_by_chunk = {}
    for chunk_id in chunk_map.keys():
        weekday_avg_by_chunk[chunk_id] = get_weekday_avg(chunk_map[chunk_id])
    return weekday_avg_by_chunk

def get_weekday_in_sub(chunk_id, pos_in_chunk, chunk_map):
    chunk = chunk_map[chunk_id]
    last = chunk[len(chunk) - 1]
    last_weekday = last[4]
    last_hour = int(last[5])
    last_pos_in_chunk = int(last[2])

    hour_diff = last_pos_in_chunk - pos_in_chunk
    if last_hour + hour_diff < 24:
        return last_weekday
    else:
        hour_diff -= 23 - last_hour
        day_diff = int(hour_diff / 24)
        weekday = last_weekday + day_diff + 1
        if weekday > 7:
            weekday -= 7
        return weekday

def ae(y, p):
    ae = 0.0
    for i in range(len(y)):
        ae += abs(float(y[i]) - p[i])
    return ae


if __name__ == '__main__':
    pass
    # res = read_file('./data/TrainingData.csv')
    # print res[0]
