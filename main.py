import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from Node import Node
from collections import Counter
from sklearn import tree



def analyze_data(data):
    #Pie chart
    plt.figure(figsize=(7, 7))
    plt.pie(x=data['target'].value_counts(), labels=data['target'].unique(), autopct="%1.1f%%")
    plt.title("Pie chart for Targer varaible")
    plt.show()

    #NA values for each column
    print(data.isnull().sum())

    #Statistical Analysis
    print(data.describe())

    #Histogram for features
    fig, ax = plt.subplots(figsize=(10, 10))
    data.hist(ax=ax)
    plt.show()

    #Box plot for features
    for col in data.columns.drop(['sex', 'fbs', 'target', 'exang']):
        plt.style.use('classic')
        plt.figure(figsize=(7, 5))
        sns.boxplot(data=data[col], orient="h")
        plt.title('Boxplot of {}'.format(col))
        plt.show()

    #Corralation matrix
    plt.figure(figsize=(7,7))
    sns.heatmap(data.corr())
    plt.title("Corralation Matrix for data")
    plt.show()



def preprocess(data):
    #fill na with mean values
    data2 = data.fillna(data.mean())

    #Perform Min Max Scaler
    data_min_max_scaled = data2.copy()
    for column in data_min_max_scaled.columns:
        data_min_max_scaled[column] = (data_min_max_scaled[column] - data_min_max_scaled[column].min()) / (
                    data_min_max_scaled[column].max() - data_min_max_scaled[column].min())

    #Split to X and y
    y = data_min_max_scaled["target"]
    X = data_min_max_scaled.drop("target", axis=1)

    return X,y


def show_roc_curve(y_true, preds):
    pass


# bonus
def calculate_class_weights(y):
    pass


if __name__ == "__main__":
    data = pd.read_csv('heart.csv')
    # analyze_data(data)
    x, y = preprocess(data)
    # X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=11)
    #
    # x_train = x.head(11)
    # y_train = y[:11]
    # n = Node(x_train, y_train)
    # print(len(n.y))

    # index1 = list(range(140))
    # index2 = list(range(140,303))
    # print(n.gini)
    # print(index1[-1],index2[0],index2[-1])
    # print(n.get_gini_gain(index1,index2))

    # sorted_x = np.sort(x['age'].unique())
    # avg_values = []
    # for i in range(len(sorted_x) - 1):
    #     avg_values.append(np.mean([sorted_x[i], sorted_x[i + 1]]))
    #
    # v = avg_values[5]
    # left_indexes = n.x[n.x['age'] < v].index.values
    # right_indexes = n.x[n.x['age'] >= v].index.values
    #
    # left_values = n.x[n.x['age'] < v]['age']
    # right_values = n.x[n.x['age'] >= v]['age']
    #
    # print(max(left_values))
    # print(min(right_values))
    # print(v)
    #
    # print(len(left_indexes) + len(right_indexes))
    #
    # n.grow_tree()

    # n.print_tree()
    # preds = n.predict(x)
    #
    # print(preds[:20])
    # print(Counter(preds))
    # print(Counter(y))

    # var1 = [0,0,0,0,1,1,1,0,1,0]
    # var2 = [33,54,56,42,50,55,31,-4,77,49]
    # y_test = [0,0,0,0,0,1,1,1,1,1]
    #
    # data = {
    #     "var1": var1,
    #     "var2": var2
    # }
    #
    # x_test = pd.DataFrame(data = data)
    # dt = tree.DecisionTreeClassifier(max_depth=5, min_samples_leaf=5)
    # dt.fit(x, y)

    n = Node(x, y)
    n.grow_tree()
    # n.print_tree()

    # plt.figure(figsize=(15, 15))
    # tree.plot_tree(dt, fontsize=7)
    # plt.show()

    preds = n.predict(x.head(10))
    print(preds)
    print(Counter(preds))























    # weights = calculate_class_weights(y)

    # implement here the experiments for task 4
