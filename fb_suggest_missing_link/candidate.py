"""
Responsible for selecting candidates.
"""

#!/usr/bin/env python

def get_surroundings(follow, followed, nodes):
    """ Gets all followers and followees of a given sets of nodes. """

    followers_and_followees = set()
    for node in nodes:
        followers_and_followees.update(follow[node])
        followers_and_followees.update(followed[node])
    return followers_and_followees

def get_candidates(follow, followed, node):
    """ Gets candidates for node to suggest follow. """

    nodes_exclude = follow[node].copy()
    nodes_exclude.add(node)

    l1_candidates = get_surroundings(follow, followed, [node])
    l2_candidates = get_surroundings(follow, followed, l1_candidates)
    l3_candidates = get_surroundings(follow, followed, l2_candidates)

    candidates = set()
    candidates.update(l1_candidates)
    candidates.update(l2_candidates)
    candidates.update(l3_candidates)

    candidates.difference_update(nodes_exclude)
    return candidates

if __name__ == '__main__':
    pass

