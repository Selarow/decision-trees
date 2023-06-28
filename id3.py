from pandas import read_csv
from numpy import log2, unique, shape
from prettyprint import PrettyPrintTree



class TreeVisualization():
    def __init__(self, value):
        self.children = []
        self.value = value

    def add_child(self, child):
        self.children.append(child)

        return child



class Node():
    def __init__(self, left=None, right=None, feature=None, threshold=None, info_gain=None, value=None):
        self.left = left
        self.right = right
        self.feature = feature
        self.threshold = threshold
        self.info_gain = info_gain
        self.value = value



class DecisionTree():
    def __init__(self, min_samples_split=3, max_depth=3):
        self.root = None
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.labels = None


    def make(self, dataset, target):
        self.target = target
        self.labels = list(dataset.columns)
        self.labels.remove(target)
        self.visual_tree = TreeVisualization("root")
        self.root = self.build_tree(dataset, self.visual_tree)


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


    def get_best_split(self, dataset):
        max_info_gain = -float("inf")
        best_split = dict()

        for feature in self.labels:
            thresholds = unique(dataset[feature])

            for t in thresholds:
                left, right = self.split(dataset, t, feature)

                if len(left) > 0 and len(right) > 0:
                    info_gain = self.information_gain(dataset.iloc[:,-1], left.iloc[:,-1], right.iloc[:,-1])

                    if info_gain > max_info_gain:
                        best_split["left"] = left
                        best_split["right"] = right
                        best_split["feature"] = feature
                        best_split["threshold"] = t
                        best_split["info_gain"] = info_gain
                        max_info_gain = info_gain
 
        return best_split


    def build_tree(self, dataset, visual_tree, depth=0):
        X = dataset.iloc[:,:-1].values
        num_samples, _ = shape(X)

        if num_samples >= self.min_samples_split and depth <= self.max_depth:
            best_split = self.get_best_split(dataset)

            if best_split["info_gain"] > 0:
                visual_child = TreeVisualization(f'{best_split["feature"]} ≤ {best_split["threshold"]}')
                visual_tree = visual_tree.add_child(visual_child)
                left_subtree = self.build_tree(best_split["left"], visual_tree, depth+1)
                right_subtree = self.build_tree(best_split["right"], visual_tree, depth+1)

                return Node(left_subtree, right_subtree, best_split["feature"], best_split["threshold"], best_split["info_gain"])

        remaining = list(dataset[self.target])
        leaf = max(remaining, key=remaining.count)
        visual_tree = visual_tree.add_child(TreeVisualization(leaf))

        return Node(value=leaf)


    def print_tree(self):
        pt = PrettyPrintTree(lambda node: node.children, lambda node: node.value)
        pt(self.visual_tree)



if __name__ == "__main__":
    def iris():
        labels = ["sepal_length", "sepal_width", "petal_length", "petal_width", "variety"]
        data = read_csv("iris.csv", skiprows=1, header=None, names=labels)
        tree = DecisionTree()
        tree.make(data, "variety")
        tree.print_tree()
    
    def heart():
        labels = ["age", "anaemia", "creatinine", "diabetes", "ejection", "high bp", "platelets", "serum", "sex", "smoking", "death"]
        data = read_csv("Heart_Failure_Details.csv", usecols=range(1, len(labels)+1), skiprows=1, header=None, names=labels)
        tree = DecisionTree()
        tree.make(data, "death")
        tree.print_tree()
    
    iris()
    #heart()
