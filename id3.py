from math import log2
from numpy import unique
from pandas import DataFrame, read_csv



class Node():
    def __init__(self, left=None, right=None, threshold=None, info_gain=None, value=None):
        self.left = left
        self.right = right
        self.threshold = threshold
        self.info_gain = info_gain
        self.value = value



class DecisionTree():
    def __init__(self, max_depth=3):
        self.root = None
        self.max_depth = max_depth
        self.labels = None


    def make(self, dataset, target):
        self.labels = list(dataset.columns)
        self.labels.remove(target)
        self.root = self.build_tree(dataset)


    def entropy(self, feature):
        entropy = 0
        n = len(feature)
        _, classes = unique(feature, return_counts=True)

        for c in classes:
            entropy += - (c / n) * log2(c / n)
        
        return entropy


    def information_gain(self, parent_feature, left_feature, right_feature):
        weight_left = len(left_feature) / len(parent_feature)
        weight_right = len(right_feature) / len(parent_feature)

        info_gain = self.entropy(parent_feature) - (weight_left * self.entropy(left_feature) + weight_right * self.entropy(right_feature))

        return info_gain


    def split(self, dataset, threshold, feature):
        left = dataset[dataset[feature] <= threshold]
        right = dataset[dataset[feature] > threshold]

        return left, right


    def get_best_split(self, dataset, feature):
        max_info_gain = -float("inf")
        thresholds = unique(dataset[feature])
        best_split = dict()

        for t in thresholds:
            left, right = self.split(dataset, t, feature)

            if len(left) > 0 and len(right) > 0:
                info_gain = self.information_gain(dataset[feature], left[feature], right[feature])

                if info_gain > max_info_gain:
                    best_split["left"] = left
                    best_split["right"] = right
                    best_split["threshold"] = t
                    best_split["info_gain"] = info_gain
                    max_info_gain = info_gain
        
        return best_split


    def build_tree(self, dataset, depth=0):
        entropies = dict()

        if depth <= self.max_depth:
            for feature in self.labels:
                entropies[self.entropy(dataset[feature])] = feature
            
            target_feature = entropies[min(entropies.keys())]

            best_split = self.get_best_split(dataset, target_feature)

            if best_split and best_split["info_gain"] > 0:
                left_subtree = self.build_tree(best_split["left"], depth+1)
                right_subtree = self.build_tree(best_split["right"], depth+1)

                return Node(left_subtree, right_subtree, best_split["threshold"], best_split["info_gain"])



if __name__ == "__main__":
    labels = ["sepal_length", "sepal_width", "petal_length", "petal_width", "variety"]
    data = read_csv("iris.csv", skiprows=1, header=None, names=labels)
    tree = DecisionTree()
    tree.make(data, "variety")
