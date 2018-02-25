import os

import pandas as pd

from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

dataset_location = 'dataset/'


def read_all_files(location):
    """
    Reads all the files train, validation, test and their corresponding classes assuming we have them all split up.

    :param location:
    :return:
    """
    df_train = pd.read_csv(location + 'features_train_data_fg.csv')
    df_validation = pd.read_csv(location + 'features_validation_data_fg.csv')
    df_test = pd.read_csv(location + 'features_test_data_fg.csv')

    df_train_class = pd.read_csv(location + 'class_train_data.csv')
    df_validation_class = pd.read_csv(location + 'class_validation_data.csv')
    df_test_class = pd.read_csv(location + 'class_test_data.csv')

    return df_train, df_train_class, df_validation, df_validation_class, df_test, df_test_class


def get_only_dirs(location):
    """
    Returns only directory names in a given location

    :param location:
    :return:
    """
    file_names = os.listdir(location)  # get all files' and folders' names in the current directory

    result = []
    for filename in file_names:  # loop through all the files and folders
        if os.path.isdir(
                os.path.join(os.path.abspath(location), filename)):  # check whether the current object is a folder
            result.append(filename)

    result.sort()
    return result


def prepare_data(train_data, train_class, validation_data, validation_class, test_data, test_class):
    """
    Prepares data for grid_search/evaluation

    :param train_data:
    :param train_class:
    :param validation_data:
    :param validation_class:
    :param test_data:
    :param test_class:
    :return:
    """
    x_train = train_data.values
    y_train = train_class.values

    x_validation = validation_data.values
    y_validation = validation_class.values

    x_test = test_data.values
    y_test = test_class.values

    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(y_train)
    encoded_y = encoder.transform(y_train)
    # convert integers to dummy variables (i.e. one hot encoded)
    y_train_encoded = np_utils.to_categorical(encoded_y)

    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(y_validation)
    encoded_y = encoder.transform(y_validation)
    # convert integers to dummy variables (i.e. one hot encoded)
    y_validation_encoded = np_utils.to_categorical(encoded_y)

    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(y_test)
    encoded_y = encoder.transform(y_test)
    # convert integers to dummy variables (i.e. one hot encoded)
    y_test_encoded = np_utils.to_categorical(encoded_y)

    return x_train, y_train_encoded, x_validation, y_validation_encoded, x_test, y_test_encoded
