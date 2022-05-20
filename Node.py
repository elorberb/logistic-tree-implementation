import pandas as pd
import numpy as np
from collections import Counter
from sklearn.linear_model import LogisticRegression


class Node:

    def __init__(self, x, y, min_leaf=5, max_depth=5, depth=0):
        self.x = x
        self.y = y
        self.min_leaf = min_leaf
        self.max_depth = max_depth
        self.depth = depth
        #count for each class in node
        self.y_counts = Counter(y)
        #Count of rows
        self.samples_count = len(y)
        #list of features
        self.features = list(self.x.columns)
        #Node's sons
        self.left = None
        self.right = None
        #Gini impurity value
        self.gini = self.gini_impurity(self.y_counts.get(0, 0), self.y_counts.get(1, 0))
        #split criteria and the feature chosen for the split
        self.split_criteria = 0
        self.best_feature = None

        #logistice reg model for leaves
        self.model = None
        self.isLeaf = False



    def grow_tree(self):

        if (self.samples_count > self.min_leaf) and (self.depth < self.max_depth) and (self.gini != 0):
            best_feature = None
            best_split = 0
            best_gain = 0
            for f in self.features:
                curr_gini_gain, split_value = self.find_best_split(f)
                if curr_gini_gain > best_gain:
                    best_feature = f
                    best_gain = curr_gini_gain
                    best_split = split_value

            self.split_criteria = best_split
            self.best_feature = best_feature

            #get x and y for left and right nodes
            left_x = self.x[self.x[best_feature] < best_split]
            right_x = self.x[self.x[best_feature] >= best_split]
            left_y = [self.y[i] for i in left_x.index.values]
            right_y = [self.y[i] for i in right_x.index.values]

            left_x.reset_index(drop=True, inplace=True)
            right_x.reset_index(drop=True, inplace=True)

            #if one of split sample size is less then min leaf build a leaf
            if len(left_y) <= self.min_leaf or len(right_y) <= self.min_leaf:
                self.model = LogisticRegression()
                self.model.fit(self.x, self.y)
                return
            #build left node
            left_node = Node(
                x=left_x,
                y=left_y,
                max_depth=self.max_depth,
                depth=self.depth + 1,
                min_leaf=self.min_leaf,
            )
            self.left = left_node
            self.left.grow_tree() #grow tree from left
            #build right node
            right_node = Node(
                x=right_x,
                y=right_y,
                max_depth=self.max_depth,
                depth=self.depth + 1,
                min_leaf=self.min_leaf,
            )
            self.right = right_node
            self.right.grow_tree() #grow tree from right


    def find_best_split(self, var_idx):

        #helper func find best split - calc avg for each two values
        def get_values_between(sorted_x):
            values_between = []
            for i in range(len(sorted_x) - 1):
                values_between.append(np.mean([sorted_x[i], sorted_x[i + 1]]))
            return values_between

        max_gini_gain = 0
        best_split = 0
        sorted_values = np.sort(self.x[var_idx].unique())
        avg_values = get_values_between(sorted_values)

        for v in avg_values:
            left_indexes = self.x[self.x[var_idx] < v].index.values
            right_indexes = self.x[self.x[var_idx] >= v].index.values

            curr_gini_gain = self.get_gini_gain(left_indexes, right_indexes)
            if curr_gini_gain > max_gini_gain:
                max_gini_gain = curr_gini_gain
                best_split = v

        return max_gini_gain, best_split

    def get_gini_gain(self, lhs, rhs):

        #get percentage for left and right
        total_len = len(lhs) + len(rhs)
        p_left = len(lhs) / total_len
        p_right = len(rhs) / total_len

        #get y values for left and right
        y_left = [self.y[i] for i in lhs]
        y_right = [self.y[i] for i in rhs]

        #calculate gini impurity for each y list
        y_left_counts = Counter(y_left)
        y_right_counts = Counter(y_right)
        gini_left = self.gini_impurity(y_left_counts.get(0, 0), y_left_counts.get(1, 0))
        gini_right = self.gini_impurity(y_right_counts.get(0, 0), y_right_counts.get(1, 0))

        #return gini gain
        gini_gain = self.gini - (p_left * gini_left + p_right * gini_right)
        return gini_gain


    def is_leaf(self):
        return self.left is None and self.right is None

    def predict(self, x):
        predictions = []
        for index, row in x.iterrows():
            predictions.append(self.predict_row(row))
        return predictions

    def predict_row(self, xi):
        while not self.is_leaf():
            if xi[self.best_feature] < self.split_criteria:
                self.left.predict_row(xi)
            else:
                self.right.predict_row(xi)

        if self.model is not None:
            return self.model.predict(xi)

        else:
            return self.y[0]


    @staticmethod
    def gini_impurity(y1_count, y2_count):
        total = y1_count + y2_count
        if total == 0: #if there is no samples
            return 0
        else: #calculate p for each class
            p1 = y1_count / total
            p2 = y2_count / total
        return 1 - (p1**2 + p2**2)


    def print_tree(self):

        print(f'node rule: {self.split_criteria} node f: {self.best_feature} y count: {Counter(self.y)} isLeaf: {self.isLeaf}')

        if self.left is not None:
            self.left.print_tree()

        if self.right is not None:
            self.right.print_tree()




