import csv
from numpy import *

def edges_generator(file_name):
    """
    Generator that returns edges given a 2-column csv graph file
    """

    f = open(file_name)
    reader = csv.reader(f)
    # Ignore the header
    reader.next()

    for edges in reader:
        nodes = [int(node) for node in edges]
        yield nodes

    f.close()

def read_graph(file_name):
    """
    Reads a sparsely represented directed graph into a dictionary
    """

    # Store the graph as a dictionary of edges
    follow = {}
    followed = {}

    def initialize_node(node):
        if node not in follow:
            follow[node] = set()
        if node not in followed:
            followed[node] = set()

    count = 0
    for nodes in edges_generator(file_name):
        for node in nodes:
            initialize_node(node)
        follow[nodes[0]].add(nodes[1])
        followed[nodes[1]].add(nodes[0])
        count += 1
        if count % 1000000 == 0:
            print 'Already read %d nodes.' % count

    return (follow, followed)

def read_file(data_file, ignore_header=True):
    """ Reads data from the file. """

    f = open(data_file)
    reader = csv.reader(f)
    if ignore_header:
        reader.next()

    data = []
    for row in reader:
        data.append(row)

    f.close()
    return data

def read_nodes_list(test_file):
    """
    Reads of single-column list of nodes
    """

    f = open(test_file)
    reader = csv.reader(f)
    reader.next()

    nodes = []
    for row in reader:
        nodes.append(int(row[0]))
    return nodes
    f.close()

def write_file(data_file, data):
    """ Writes the data to the data_file. """

    f = open(data_file, 'w')
    writer = csv.writer(f)
    for row in data:
        writer.writerow(row)
    f.close()

def write_submission_file(submission_file, test_nodes, test_predictions):
    """
    Writes the submission file
    """

    f = open(submission_file, "w")
    writer = csv.writer(f)
    writer.writerow(["source_node", "destination_nodes"])

    for source_node, dest_nodes in zip(test_nodes, test_predictions):
        writer.writerow([str(source_node),
                         " ".join([str(n) for n in dest_nodes])])
    f.close()

