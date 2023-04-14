import sys

import matplotlib
import numpy
import pandas
from matplotlib import pyplot as plt
from pandas import DataFrame
import scipy
import sklearn
import missingno as mi
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.utils.multiclass import type_of_target


def load_dataset():
    # load the dataset using pandas library
    titanic_dataset: DataFrame | None = pandas.read_csv("titanic_dataset.csv")

    # create Boolean mask indicating where missing values are located
    missing_values_mask = titanic_dataset.isnull()

    # count number of missing values for each column
    missing_values_count = missing_values_mask.sum()

    # visualize missing data in the dataframe
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    mi.matrix(titanic_dataset, ax=axs[0], sparkline=False)
    axs[0].set_title("Before Imputation")

    # print count of missing values in each column
    print(missing_values_count)

    # define a function to map 'male' to 0 and 'female' to 1
    def map_gender(x):
        if x == "male":
            return 0
        elif x == "female":
            return 1

    # apply the function to the 'gender' column of the dataframe
    titanic_dataset["sex"] = titanic_dataset["sex"].apply(map_gender)

    # create a SimpleImputer object with strategy='median'
    imputer = SimpleImputer(strategy="mean")

    median_list_imputation = ["age", "fare"]

    # replace NaN values in column 'age' with median value
    titanic_dataset[median_list_imputation] = imputer.fit_transform(titanic_dataset[median_list_imputation])

    # create a list of column labels that will be used for imputation of data with 0 (will not be used in the analysis)
    imputation_0 = ["cabin", "embarked", "boat", "body", "home.dest"]

    # replace missing values in columns "cabin", "embarked", "boat", "body", "home.dest" with 0
    # 0 was used as there is no way to get accurate data. However, because filling in the missing values with 0 would
    # cause invalid results, every column which was imputed with 0 will not be used.
    titanic_dataset[imputation_0] = titanic_dataset[imputation_0].fillna(0)

    # create Boolean mask indicating where missing values are located
    missing_values_mask_ctrl = titanic_dataset.isnull()

    # count number of missing values for each column
    missing_values_count_ctrl = missing_values_mask_ctrl.sum()

    # visualize missing data in the dataframe after imputation
    mi.matrix(titanic_dataset, ax=axs[1], sparkline=False)
    axs[1].set_title("After Imputation")

    # adjust the layout of the subplots to make room for the titles
    fig.subplots_adjust(top=0.7, bottom=0.1)

    plt.show()

    # print count of missing values in each column
    print(missing_values_count_ctrl)

    return titanic_dataset


def split_dataset(titanic_dataset):
    # create a numpy array with the values of the pandas DataFrame
    titanic_array = pandas.DataFrame(titanic_dataset).values

    # # variable representing selected columns for the training input features
    selected_columns = ["pclass", "survived", "sex", "age"]

    # convert column names to indices
    selected_indices = [titanic_dataset.columns.get_loc(col) for col in selected_columns]

    # print the data to check if they correspond to the data in csv file
    print(selected_columns)
    print(titanic_array[0:4, selected_indices])

    # create variable containing input features
    input_features = titanic_array[:, selected_indices]
    print("This is the input data\n", input_features)

    target_type = type_of_target(input_features)
    print("This is the target type:", target_type)

    # create a variable containing output labels
    output_labels = titanic_array[:, 1]

    for i in range(len(output_labels)):
        if output_labels[i] == 1:
            output_labels[i] = "survived"
        elif output_labels[i] == 0:
            output_labels[i] = "did not survive"
        else:
            print("not 0 or 1")

    print("This is the output labels\n", output_labels)

    target_type = type_of_target(output_labels)
    print("This is the target type:", target_type)

    train_input_features, validate_input_features, train_output_labels, validate_output_labels = train_test_split(
        input_features, output_labels, test_size=0.30, random_state=1, shuffle=True)

    return train_input_features, validate_input_features, train_output_labels, validate_output_labels


def test_models(train_input_features, train_output_labels):
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
        cv_results = cross_val_score(model, train_input_features, train_output_labels, cv=kfold, scoring='accuracy')

        # append cv_results to the empty list results
        results.append(cv_results)

        # append name to the empty list names
        names.append(name)

        # print the name of the model, mean value of cv_results and average value of cv_results
        print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

    # return the variables of the function
    return models, results, names, cv_results


def run_the_program():
    titanic_dataset = load_dataset()
    train_input_features, validate_input_features, train_output_labels, validate_output_labels = \
        split_dataset(titanic_dataset)
    models, results, names, cv_results = test_models(train_input_features, train_output_labels)

    pass


run_the_program()
