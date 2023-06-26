from math import log2
from numpy import unique
from pandas import DataFrame, read_csv



labels = ["sepal_length", "sepal_width", "petal_length", "petal_width", "variety"]
data = read_csv("iris.csv", skiprows=1, header=None, names=labels)


def entropy(feature):
    entropy = 0
    n = len(feature)
    _, classes = unique(feature, return_counts=True)

    for c in classes:
        entropy += - (c / n) * log2(c / n)
    
    return entropy


def information_gain(parent_feature, left_feature, right_feature):
    weight_left = len(left_feature) / len(parent_feature)
    weight_right = len(right_feature) / len(parent_feature)

    info_gain = entropy(parent_feature) - (weight_left * entropy(left_feature) + weight_right * entropy(right_feature))

    return info_gain


def split(dataset, threshold, feature):
    left = dataset[dataset[feature] <= threshold]
    right = dataset[dataset[feature] > threshold]

    return left, right


def get_best_split(dataset, feature):
    max_info_gain = -float("inf")
    thresholds = unique(dataset[feature])
    best_split = dict()

    for t in thresholds:
        left, right = split(dataset, t, feature)

        if len(left) > 0 and len(right) > 0:
            info_gain = information_gain(dataset[feature], left[feature], right[feature])

            if info_gain > max_info_gain:
                best_split["threshold"] = t
                best_split["left"] = left
                best_split["right"] = right
                best_split["info_gain"] = info_gain
                max_info_gain = info_gain
    
    return best_split


def build_tree(dataset):
    pass






#######################
##      TESTING      ##
#######################

labels = labels[:len(labels) - 1]
entropies = dict()

for feature in labels:
    entropies[entropy(data[feature])] = feature

target_feature = entropies[min(entropies.keys())]
best_split = get_best_split(data, target_feature)
print(best_split)
