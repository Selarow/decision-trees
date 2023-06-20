from math import log2
from bigtree import Node, print_tree
from pandas import Series, read_csv, value_counts



col_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "variety"]
data = read_csv("iris.csv", skiprows=1, header=None, names=col_names)


def get_entropy(dataset, target):
    entropy = 0
    n = len(dataset)
    target_data = Series(dataset.loc[:, target]).value_counts()
    classes = target_data.values

    for c in classes:
        entropy += - (c / n) * log2(c / n)
    
    return entropy


def information_gain(dataset, feature):
    pass
