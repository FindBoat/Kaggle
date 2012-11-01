"""
This script is responsible for generating test data and analyzing the
prediction result.
"""

#!/usr/bin/env python
import utilities
import candidate
from numpy import *
from random import randint
import rank

def generate_test_nodes(follow, nodes_exclude, num):
    """ Generates nodes for test. """

    test_nodes = []
    nodes_all = list(follow.keys())
    perm = random.permutation(len(nodes_all))
    for index in perm:
        node = nodes_all[index]
        if len(follow[node]) > 4 and node not in nodes_exclude:
            test_nodes.append(node)
            if len(test_nodes) >= num:
                break
    return test_nodes

def remove_edges(follow, followed, node, max_remove_num):
    """ Randomly remove edges for the given node. The dict follow
    and follwed are modified in this method"""

    followees = list(follow[node])
    num_followees = len(followees)
    r = randint(4, min(num_followees, max_remove_num))
    perm = random.permutation(num_followees)
    perm = perm[0 : r]

    edges_removed = []
    for index in perm:
        n = followees[index]
        follow[node].remove(n)
        followed[n].remove(node)
        edges_removed.append(n)
    return edges_removed

def generate_solution(follow, followed, nodes_test, max_remove_num):
    """ Generates the solution for suggestion missing links. """

    solution = []
    for node in nodes_test:
        s = [node]
        edges_removed = remove_edges(follow, followed, node, max_remove_num)
        s += edges_removed
        solution.append(s)
    return solution

def generate_test_set(follow, followed, test_file, validation_file,
    solution_file, num,  max_remove_num):
    """ Generates the test set for analysis. """

    nodes_exclude = utilities.read_nodes_list(test_file)

    print 'Generating test nodes...'
    nodes_test = generate_test_nodes(follow, nodes_exclude, num)
    writable_nodes_test = [[n] for n in nodes_test]
    solution = generate_solution(follow, followed, nodes_test, max_remove_num)

    utilities.write_file(validation_file, writable_nodes_test)
    utilities.write_file(solution_file, solution)

def generate_training_set(follow, followed, ratio, solution_file, data_file):
    """ Uses the solution file to generate training set to train
    the model, hoping this method can get better result.
    Ratio controls the fraction of pos and neg data sets, if ratio is -1,
    the fraction is the origion fraction."""

    raw_solution = utilities.read_file(solution_file, False)
    dict_solution = {}
    for i in range(len(raw_solution)):
        row = raw_solution[i]
        dict_solution[int(row[0])] = set(int(n) for n in row[1 : :])

    x_train = [['spring brother is a true man']]
    for node in dict_solution.keys():
        nodes_pos = dict_solution[node]
        for n in nodes_pos:
            features = rank.get_features(follow, followed, node, n)
            x_train.append([1] + features)

        nodes_neg = candidate.get_candidates(follow, followed, node)
        nodes_neg.difference_update(nodes_pos)
        nodes_neg = list(nodes_neg)
        perm = random.permutation(len(nodes_neg))
        if ratio != -1:
            num = min(int(len(nodes_pos) * ratio), len(nodes_neg))
        else:
            num = len(nodes_neg)
        for i in range(num):
            node = nodes_neg[perm[i]]
            features = rank.get_features(follow, followed, node, n)
            x_train.append([0] + features)

    utilities.write_file(data_file, x_train)

def analyze_candidates(solution_file, follow, followed):
    """ Analyzes the method get_candidates. """

    raw_solution = utilities.read_file(solution_file, False)
    dict_solution = {}
    for row in raw_solution:
        dict_solution[int(row[0])] = set(int(n) for n in row[1 : :])

    count_total = 0
    count_miss = 0
    for node in dict_solution:
        candidates = candidate.get_candidates(follow, followed, node)
        for n in dict_solution[node]:
            if n not in candidates:
                count_miss += 1
        count_total += len(dict_solution[node])

    print 'count_total = %d, count_miss = %d' %(
        count_total, count_miss)

def ap(ground_truth, prediction):
    """ Calculates the average precision. """

    ap = 0.0
    already_hit = 0
    for i in range(len(prediction)):
        if prediction[i] in ground_truth:
            already_hit += 1
            ap += 1.0 * already_hit / (i + 1)
    ap /= len(ground_truth)
    return ap

def mean_average_precision(result_file, solution_file):
    """ Calculates the mean average precision. """

    raw_result = utilities.read_file(result_file, True)
    raw_solution = utilities.read_file(solution_file, False)
    dict_result = {}
    for row in raw_result:
        dict_result[row[0]] = row[1 : :]
    dict_solution = {}
    for row in raw_solution:
        dict_solution[row[0]] = set(row[1 : :])

    res = 0.0
    for key in dict_result.keys():
        prediction = dict_result[key][0].split()
        ground_truth = dict_solution[key]
        res += ap(ground_truth, prediction)
    res /= len(dict_result)
    print 'mean average precision = %f' % res

if __name__ == '__main__':
    mean_average_precision('./data/result.csv',
        './data/solution.csv')

    # follow, followed = utilities.read_graph('./data/train.csv')
    # generate_test_set(follow, followed,
    #      './data/test.csv',
    #      './data/validation.csv',
    #      './data/solution.csv',
    #      10000, 10)
    # print 'Generating training set...'
    # generate_training_set(follow, followed, 10,
    #     './data/solution.csv',
    #     './data/data.csv')

    # print 'Generating test set...'
    # generate_training_set2(follow, followed, -1,
    #     './data/solution.csv',
    #     './data/data_test.csv')

#    analyze_candidates('./data/solution.csv', follow, followed)





