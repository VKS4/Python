# check package versions
import sys

import matplotlib
import numpy
import pandas
import scipy
import sklearn
from matplotlib import pyplot
from pandas import DataFrame
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

print('Python: {}'.format(sys.version))
print('scipy: {}'.format(scipy.__version__))
print('numpy: {}'.format(numpy.__version__))
print('matplotlib: {}'.format(matplotlib.__version__))
print('pandas: {}'.format(pandas.__version__))
print('sklearn: {}'.format(sklearn.__version__))


def load_dataset():
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    iris_dataset: DataFrame | None = pandas.read_csv("iris.csv")

    # print how many instances(rows) and attributes(columns) does the data set have
    print(iris_dataset.shape)

    # print the first 20 lines of data of all columns
    print(iris_dataset.head(20))

    # print statistic information about the dataset
    print(iris_dataset.describe())

    return names, iris_dataset


def plot_dataset(iris_dataset):
    # box and whisker plots
    iris_dataset.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
    pyplot.show()

    # histograms
    iris_dataset.hist()
    pyplot.show()

    # scatter plot matrix
    pandas.plotting.scatter_matrix(iris_dataset)
    pyplot.show()

    array = iris_dataset.values
    x = array[:, 0:4]
    y = array[:, 4]

    return x, y, array


def split_test(x, y, array):
    print('Length of array X', len(x))
    print('Length of array Y', len(y))

    # split the dataset into 20% and 80%. (learning dataset part and testing dataset part)
    a = len(x) * 0.2
    b = len(y) * 0.2

    a2 = len(x) * 0.8
    b2 = len(y) * 0.8

    learn_x = array[0:int(a2), 0:4]
    learn_y = array[0:int(b2), 4]

    test_x = array[0:int(a), 0:4]
    test_y = array[0:int(b), 4]

    print('Test X array ', test_x)
    print('Test Y array ', test_y)

    print('Learn X array ', learn_x)
    print('Learn Y array ', learn_y)

    return learn_x, learn_y, test_x, test_y


def test_models(learn_x, learn_y):
    models = [('LR', LogisticRegression(solver='liblinear', multi_class='ovr')),
              ('LDA', LinearDiscriminantAnalysis()),
              ('KNN', KNeighborsClassifier()),
              ('CART', DecisionTreeClassifier()),
              ('NB', GaussianNB()),
              ('SVM', SVC(gamma='auto'))]

    results = []
    names = []
    cv_results = []

    for name, model in models:
        kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
        cv_results = cross_val_score(model, learn_x, learn_y, cv=kfold, scoring='accuracy')
        results.append(cv_results)
        names.append(name)

        print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

    return models, results, names, cv_results


names_out, iris_dataset_out = load_dataset()
X, Y, array = plot_dataset(iris_dataset_out)
learn_x, learn_y, test_x, test_y = split_test(X, Y, array)
models, results, names, cv_results = test_models(learn_x, learn_y)
