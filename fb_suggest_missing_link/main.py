"""
Based on number of mutual friends - 15%
Based on number of mutual follows - 30%
Remove non-followed suggestion - 66.3%
"""

#!/usr/bin/env python
from collections import deque
import utilities
import time
import rank
import candidate
import validation

def get_popular_people(followed, num):
    """ Gets people with most followers. """

    dict_num_followers = {}
    for node in followed.keys():
        dict_num_followers[node] = len(followed[node])

    popular_people = sorted(dict_num_followers,
        key=dict_num_followers.__getitem__,
        reverse=True)

    return popular_people[0 : num]

def suggest_friends(follow, followed, clf, node, popular_people,
    max_suggestion):
    """ Suggests friends for a given node. """

    if not follow.has_key(node):
        return []

    candidates = candidate.get_candidates(follow, followed, node)
    suggested = rank.rank_candidates(follow, followed, clf, node, candidates)

    # Suggests most popular people when candidates are less than 10.
    if len(suggested) < max_suggestion:
        for star in popular_people:
            if star not in suggested:
                suggested.append(star)
            if len(suggested) >= max_suggestion:
                break
    else:
        suggested = suggested[0 : max_suggestion]

    return suggested

def main(follow, followed, test_file, submission_file, data_file,
    validation_file, max_suggestion):
    """ The main method for the problem. """

    print 'Reading graph...'
    test_nodes = utilities.read_nodes_list(test_file)

    print 'Training with logistic regression...'
    clf = rank.train(data_file, validation_file)

    print 'Getting popular people...'
    popular_people = get_popular_people(followed, max_suggestion)

    print 'Predicting...'
    predictions = []
    count = 0
    for node in test_nodes:
        suggested = suggest_friends(follow, followed, clf, node,
            popular_people, max_suggestion)
        predictions.append(suggested)

        count += 1
        if count % 100 == 0:
            print 'Suggested %d friends.' % count

    print 'Writing submission files...'
    utilities.write_submission_file(submission_file, test_nodes, predictions)

if __name__ == '__main__':
    start_time = time.time()
    follow, followed = utilities.read_graph('./data/train.csv')

    validation.generate_test_set(follow, followed,
         './data/test.csv',
         './data/validation.csv',
         './data/solution.csv',
         2000, 10)

    main(follow, followed,
         './data/validation.csv',
         './data/result.csv',
         './data/data.csv',
         './data/data_test.csv',
         10)

    # main(follow, followed,
    #      './data/test.csv',
    #      './data/result.csv',
    #      './data/data.csv',
    #      10)

    print (time.time() - start_time) / 60.0, 'minutes'

