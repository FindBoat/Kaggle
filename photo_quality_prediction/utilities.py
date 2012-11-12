#!/usr/bin/env python
import csv
from numpy import *

def write_submission_file(file_name, ids, p):

    print 'Writing submission file...'
    f = open(file_name, 'w')
    writer = csv.writer(f)
    for i in range(len(p)):
        writer.writerow([ids[i], p[i]])
    f.close()

def write_file(file_name, data, single=True):
    """ Writes data to the file. """

    print 'Writing output file...'
    f = open(file_name, 'w')
    writer = csv.writer(f)
    for line in data:
        if single:
            writer.writerow([line])
        else:
            writer.writerow(line)
    f.close()

def read_file(file_name, single=True):
    """ Reads file. """

    print 'Reading file...'
    f = open(file_name)
    reader = csv.reader(f)
    res = []
    for line in reader:
        if single:
            res.append(line[0])
        else:
            res.append(line)

    f.close()
    return res

def read_test_file(file_name):
    print 'Reading test file...'
    f = open(file_name)
    reader = csv.reader(f)
    reader.next()

    ids = []
    meta_data = []
    for line in reader:
        latitude = int(line[1])
        longitude = int(line[2])
        width = int(line[3])
        height = int(line[4])
        size = int(line[5])
        name = line[6]
        description = line[7]
        caption = line[8]

        ids.append(line[0])
        meta_data.append([latitude, longitude, width, height, size, name,
            description, caption])

    f.close()
    return (ids, meta_data)

def read_training_file(file_name):
    """ Reads training file and generates data. """

    print 'Reading training file...'
    f = open(file_name)
    reader = csv.reader(f)
    reader.next()

    y = []
    meta_data = []
    for line in reader:
        latitude = int(line[1])
        longitude = int(line[2])
        width = int(line[3])
        height = int(line[4])
        size = int(line[5])
        name = line[6]
        description = line[7]
        caption = line[8]
        good = int(line[9])

        y.append(good)
        meta_data.append([latitude, longitude, width, height, size, name,
            description, caption])

    f.close()
    return (y, meta_data)

def sample(y, meta_data, num, randomly=True):
    """ Randomly samples num data from the whole data set. """

    if num == -1:
        num = len(y)
    y_sample = []
    meta_data_sample = []
    perm = range(len(y))
    if randomly:
        perm = random.permutation(len(y))
    perm = perm[0 : min(num, len(y))]
    for index in perm:
        y_sample.append(y[index])
        meta_data_sample.append(meta_data[index])
    return (y_sample, meta_data_sample)

def binomial_deviance(y, prediction):
    """ Calculates the binomial deviance for the prediction. """

    binomial_deviance = 0.0
    for i in range(len(prediction)):
        if prediction[i] > .99:
            prediction[i] = .99
        elif prediction[i] < .1:
            prediction[i] = .1
        tmp = y[i] * math.log10(prediction[i])
        tmp += (1 - y[i]) * math.log10(1 - prediction[i])
        binomial_deviance -= tmp
    binomial_deviance /= float(len(prediction))
    return binomial_deviance

if __name__ == '__main__':
    pass
