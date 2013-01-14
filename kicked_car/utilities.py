#!/usr/bin/env python
import csv
from numpy import *

def read_file(file_name, header=False):
    print 'Reading file...'
    f = open(file_name)
    reader = csv.reader(f)
    if not header:
        reader.next()
    res = [line for line in reader]
    f.close()
    return res

def write_submission_file(file_name, refid, predict):
    print 'Writing submission file...'
    f = open(file_name, 'w')
    writer = csv.writer(f)
    for i in range(len(refid)):
        writer.writerow([refid[i], predict[i][1]])
    f.close()

if __name__ == '__main__':
    pass
