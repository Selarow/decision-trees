from math import log2
from numpy import unique
from pandas import read_csv



labels = ["sepal_length", "sepal_width", "petal_length", "petal_width", "variety"]
data = read_csv("iris.csv", skiprows=1, header=None, names=labels)


def entropy(feature):
    entropy = 0
    n = len(feature)
    _, classes = unique(feature, return_counts=True)

    for c in classes:
        entropy += - (c / n) * log2(c / n)
    
    return entropy


def information_gain(parent, left, right):
    weight_left = len(left) / len(parent)
    weight_right = len(right) / len(parent)

    info_gain = entropy(parent) - (weight_left * entropy(left) + weight_right * entropy(right))

    return info_gain


def split(dataset, threshold, feature):
    left = []
    right = []

    for i in dataset[feature]:
        if i <= threshold:
            left.append(i)
        
        else:
            right.append(i)
    
    return left, right


def get_best_split(dataset, feature):
    max_info_gain = -float("inf")
    thresholds = unique(dataset[feature])
    best_split = dict()

    for t in thresholds:
        left, right = split(dataset, t, feature)

        if len(left) > 0 and len(right) > 0:
            info_gain = information_gain(dataset[feature], left, right)

            if info_gain > max_info_gain:
                best_split["threshold"] = t
                best_split["left"] = left
                best_split["right"] = right
                best_split["info_gain"] = info_gain
                max_info_gain = info_gain
    
    return best_split






#######################
##      TESTING      ##
#######################

labels = labels[:len(labels) - 1]
entropies = dict()

for feature in labels:
    entropies[entropy(data[feature])] = feature

target_feature = entropies[min(entropies.keys())]

print(get_best_split(data, target_feature))
