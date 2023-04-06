# import libraries
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

# check package versions
print("Python: {}".format(sys.version))
print("scipy: {}".format(scipy.__version__))
print("numpy: {}".format(numpy.__version__))
print("matplotlib: {}".format(matplotlib.__version__))
print("pandas: {}".format(pandas.__version__))
print("sklearn: {}".format(sklearn.__version__))


# declaration of a function that loads up the dataset
def load_dataset():
    # create a list of names of the features of the iris flowers
    names = ["sepal-length", "sepal-width", "petal-length", "petal-width", "class"]

    # load the dataset from local file iris.csv using the pandas library
    iris_dataset: DataFrame | None = pandas.read_csv("iris.csv")

    # print how many rows and columns does the dataset have
    print(iris_dataset.shape)

    # print the first 20 lines of data of all columns
    print(iris_dataset.head(20))

    # print statistic information about the dataset
    print(iris_dataset.describe())

    # return the variables from the function
    return names, iris_dataset


# declaration of a function that plots the data from the dataset
def plot_dataset(iris_dataset):
    # create box and whisker plots
    iris_dataset.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
    pyplot.show()

    # create histograms
    iris_dataset.hist()
    pyplot.show()

    # create scatter plot matrix
    pandas.plotting.scatter_matrix(iris_dataset)
    pyplot.show()

    # use pandas libray to extract values from the iris_dataset and load them up to dataset variable as a numpy array
    dataset = iris_dataset.values

    # assign values from the dataset array to the variable x (all rows, and first 4 columns (0 to 3))
    x = dataset[:, 0:4]

    # assign values from dataset to the variable y (all rows, 3rd column)
    y = dataset[:, 4]

    # return the variables from the function
    return x, y, dataset


# declaration of a function that splits the dataset into training and validation part
def split_dataset(x, y, dataset):
    # print the length of array x
    print('Length of array X', len(x))

    # print the length of array y
    print('Length of array Y', len(y))

    # use pandas library to create a new numpy array from the dataset array with included metadata
    dataset_array = pandas.DataFrame(dataset).values

    # assign values to the variable x from the array dataset_array (all rows, first 4 columns(first index is inclusive,
    # stop index is exclusive))
    x = dataset_array[:, 0:4]

    # assign values to the variable y from the array dataset_array (all rows, 5th column(4th index))
    # contains the names of the flowers (used to check accuracy of the model)
    y = dataset_array[:, 4]

    # split the x and y arrays into training (80%) and validation (20%) sets, set that the same sequence of data is used
    # always, shuffles the dataset before splitting
    train_x, validation_x, train_y, validation_y = train_test_split(x, y, test_size=0.20, random_state=1, shuffle=True)

    # return the variables from the function
    return train_x, validation_x, train_y, validation_y


# declaration of a function that will test the accuracy of models
def test_models(train_x, train_y):
    # loads the machine learning models into memory as a list of tuples
    models = [('LR', LogisticRegression(solver='liblinear', multi_class='ovr')),
              ('LDA', LinearDiscriminantAnalysis()),
              ('KNN', KNeighborsClassifier()),
              ('CART', DecisionTreeClassifier()),
              ('NB', GaussianNB()),
              ('SVM', SVC(gamma='auto'))]

    # declares empty lists results, names and cv_results
    results = []
    names = []
    cv_results = []

    # unpacking of the list of tuples, name will be assigned first value from the tuple (string with the name of the
    # model) and model will be assigned the second value in the tuple (the model itself)
    for name, model in models:
        # use kfold cross-validation to evaluate the performance of the models
        kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)

        # use cross validation function from the scikit-learn library to evaluate the models
        # new model for each iteration
        cv_results = cross_val_score(model, train_x, train_y, cv=kfold, scoring='accuracy')

        # append cv_results to the empty list results
        results.append(cv_results)

        # append name to the empty list names
        names.append(name)

        # print the name of the model, mean value of cv_results and average value of cv_results
        print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

    # return the variables of the function
    return models, results, names, cv_results


# declaration of a function for making predictions based on the trained data
def make_predictions():
    # creating an instance of the function load_order
    names_out, iris_dataset_out = load_dataset()

    # creating an instance of the function plot_dataset
    X, Y, array = plot_dataset(iris_dataset_out)

    # creating an instance of the function split_dataset
    train_x, validation_x, train_y, validation_y = split_dataset(X, Y, array)

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
    # creating an instance of the function load_dataset
    names, iris_dataset = load_dataset()

    # creating an instance of the function plot_dataset
    X, Y, array = plot_dataset(iris_dataset)

    # creating an instance of the function split_dataset
    train_x, validation_x, train_y, validation_y = split_dataset(X, Y, array)

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

    pass


# condition, if the name of the file is main, then execute function main
if __name__ == '__main__':
    main()
