import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from Node import Node
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from LogisticModelTree import LogisticModelTree
from collections import Counter


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
    #fill NA with mean values
    num_f = ['oldpeak', 'age', 'trestbps', 'chol', 'fbs', 'thalach']
    cat_f = ['thal', 'sex', 'restecg', 'exang', 'slope', 'ca']

    for c_f in cat_f:
        data[c_f].fillna(data[c_f].mode()[0], inplace=True)

    for n_f in num_f:
        data[n_f].fillna(data[n_f].mean(), inplace=True)

    #Perform Min Max Scaler
    data_min_max_scaled = data.copy()
    for column in data_min_max_scaled.columns:
        data_min_max_scaled[column] = (data_min_max_scaled[column] - data_min_max_scaled[column].min()) / (
                    data_min_max_scaled[column].max() - data_min_max_scaled[column].min())

    #Split to X and y
    y = data_min_max_scaled["target"]
    X = data_min_max_scaled.drop("target", axis=1)

    return X,y


def show_roc_curve(y_true, preds):
    auc = roc_auc_score(y_true, preds)
    print(' AUC score =%.3f' % (auc))
    ns_fpr, ns_tpr, _ = roc_curve(y_true, preds)
    plt.plot(ns_fpr, ns_tpr, marker='.')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.show()

# bonus
def calculate_class_weights(y):
    pass


if __name__ == "__main__":
    ########################################### - plotting 3 ROC curves with diffrent parameters in the same plot
    data = pd.read_csv('heart.csv')
    # analyze_data(data)
    x, y = preprocess(data)
    trainX, testX, trainy, testy = train_test_split(x, y, test_size=0.5, random_state=12)
    ns_probs = [0 for _ in range(len(testy))]
    n1 = Node(x, y, min_leaf=3, max_depth=3)
    n2 = Node(x, y, min_leaf=5, max_depth=5)
    n3 = Node(x, y, min_leaf=8, max_depth=8)
    n1.grow_tree()
    n2.grow_tree()
    n3.grow_tree()

    lr_probs = n1.predict(testX)
    lr_probs1 = n2.predict(testX)
    lr_probs2 = n3.predict(testX)

    ns_auc = roc_auc_score(testy, ns_probs)
    lr_auc = roc_auc_score(testy, lr_probs)
    lr_auc1 = roc_auc_score(testy, lr_probs1)
    lr_auc2 = roc_auc_score(testy, lr_probs2)
    print('No Skill: ROC AUC=%.3f' % (ns_auc))
    print('Logistic regression with min_leaf=3, max_depth=3: ROC AUC=%.3f' % (lr_auc))
    print('Logistic regression with min_leaf=5, max_depth=5: ROC AUC=%.3f' % (lr_auc1))
    print('Logistic regression with min_leaf=8, max_depth=8: ROC AUC=%.3f' % (lr_auc2))
    ns_fpr, ns_tpr, _ = roc_curve(testy, ns_probs)
    plr_fpr, lr_tpr, _ = roc_curve(testy, lr_probs)
    plr_fpr1, lr_tpr1, _ = roc_curve(testy, lr_probs1)
    plr_fpr2, lr_tpr2, _ = roc_curve(testy, lr_probs2)
    # plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(plr_fpr, lr_tpr, marker='.', label='ROC- min_leaf=3, max_depth=3 (area = 0.874)')
    plt.plot(plr_fpr1, lr_tpr1, marker='.', label='ROC- min_leaf=5, max_depth=5 (area = 0.94)')
    plt.plot(plr_fpr2, lr_tpr2, marker='.', label='ROC- min_leaf=8, max_depth=8 (area = 0.928)')
    plt.title('Receiver operating characteristic')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    # show the plot
    plt.show()
























    # weights = calculate_class_weights(y)

    # implement here the experiments for task 4
