import sys

import matplotlib
import numpy
import pandas
from pandas import DataFrame
import scipy
import sklearn
from sklearn.model_selection import train_test_split


def load_dataset():
    # load the dataset using pandas library
    titanic_dataset: DataFrame | None = pandas.read_csv("titanic_dataset.csv")

    # print first 10 rows of the dataset
    print(titanic_dataset.head(10))

    return titanic_dataset


def split_dataset(titanic_dataset):
    # create a numpy array with the values of the pandas DataFrame
    titanic_array = pandas.DataFrame(titanic_dataset).values

    # variable representing selected columns for the training input features
    selected_columns = ["pclass", "survived", "sex", "age", "boat"]

    # convert column names to indices
    selected_indices = [titanic_dataset.columns.get_loc(col) for col in selected_columns]

    # print the data to check if they correspond to the data in csv file
    print(selected_columns)
    print(titanic_array[0:4, selected_indices])

    # create variable containing input features
    input_features = titanic_array[:, selected_indices]

    # create a variable containing output labels
    output_labels = titanic_array[:, 1]

    train_input_features, validate_input_features, train_output_labels, validate_output_labels = train_test_split(
        input_features, output_labels, test_size=0.30, random_state=1, shuffle=True)

    return train_input_features, validate_input_features, train_output_labels, validate_output_labels


def run_the_program():
    titanic_dataset = load_dataset()
    train_input_features, validate_input_features, train_output_labels, validate_output_labels = \
        split_dataset(titanic_dataset)

    pass


run_the_program()
