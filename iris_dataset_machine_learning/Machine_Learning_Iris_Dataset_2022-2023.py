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
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split
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

    dataset = iris_dataset.values
    x = dataset[:, 0:4]
    y = dataset[:, 4]

    return x, y, dataset


def split_test(x, y, dataset):
    print('Length of array X', len(x))
    print('Length of array Y', len(y))

    dataset_array = pandas.DataFrame(dataset).values
    X = dataset_array[:, 0:4]
    y = dataset_array[:, 4]
    train_x, validation_x, train_y, validation_y = train_test_split(X, y, test_size=0.20, random_state=1, shuffle=True)

    return train_x, validation_x, train_y, validation_y


def test_models(train_x, train_y):
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
        cv_results = cross_val_score(model, train_x, train_y, cv=kfold, scoring='accuracy')
        results.append(cv_results)
        names.append(name)

        print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

    return models, results, names, cv_results


def make_predictions(models, results, names, cv_results):
    names_out, iris_dataset_out = load_dataset()
    X, Y, array = plot_dataset(iris_dataset_out)
    train_x, validation_x, train_y, validation_y = split_test(X, Y, array)
    models, results, names, cv_results = test_models(train_x, train_y)

    # Make predictions on validation dataset
    model = SVC(gamma='auto')
    model.fit(train_x, train_y)
    predictions = model.predict(validation_x)

    # Evaluate predictions
    print(accuracy_score(validation_y, predictions))
    print(confusion_matrix(validation_y, predictions))
    print(classification_report(validation_y, predictions))

    pass


def main():
    # Load the dataset
    names, iris_dataset = load_dataset()

    # Plot the dataset
    X, Y, array = plot_dataset(iris_dataset)

    # Split the dataset into training and validation sets
    train_x, validation_x, train_y, validation_y = split_test(X, Y, array)

    # Test different models on the training set
    models, results, names, cv_results = test_models(train_x, train_y)

    # Make predictions on the validation set using the best model
    model = SVC(gamma='auto')
    model.fit(train_x, train_y)
    predictions = model.predict(validation_x)

    # Evaluate the predictions
    print(accuracy_score(validation_y, predictions))
    print(confusion_matrix(validation_y, predictions))
    print(classification_report(validation_y, predictions))


if __name__ == '__main__':
    main()
