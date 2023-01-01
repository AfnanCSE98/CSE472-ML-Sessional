
import numpy as np
import pandas as pd

def load_dataset(fname):
    """
    read data_banknote_authentication.csv file
    and create 2D feature matrix X from the first four columns
    and 1D label vector y from the last column
    :return: X,y
    """
    # read data_banknote_authentication.csv file and skip the first row
    data = pd.read_csv(fname, header=None, skiprows=1)
    # create 2D feature matrix X from the first four columns
    X = data.iloc[:, :-1].values

    # create 1D label vector y from the last column
    y = data.iloc[:, -1].values

    # perform feature scaling on X
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    return X, y


def split_dataset(X, y, test_size = 0.2, shuffle = True):
    """
    function for spliting dataset into train and test
    :param X:
    :param y:
    :param float test_size: the proportion of the dataset to include in the test split
    :param bool shuffle: whether to shuffle the data before splitting
    :return:
    """
    # todo: implement.
    X_train, y_train, X_test, y_test = None, None, None, None

    if shuffle:
        # shuffle indices
        idx = np.random.permutation(range(len(y)))

        # shuffle together
        X, y = X[idx], y[idx]
    
    # split data
    X_train, X_test = X[:int(len(y) * (1 - test_size))], X[int(len(y) * (1 - test_size)):]

    # split labels
    y_train, y_test = y[:int(len(y) * (1 - test_size))], y[int(len(y) * (1 - test_size)):]

    # assert shapes
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]


    return X_train, y_train, X_test, y_test


def bagging_sampler(X, y):
    """
    Randomly sample with replacement
    Size of sample will be same as input data
    :param X:
    :param y:
    :return:
    """
    # todo: implement
    X_sample, y_sample = None, None
    # randomly sample X and y with replacement
    idx = np.random.choice(range(len(y)), size=len(y), replace=True)
    X_sample, y_sample = X[idx], y[idx]

    # assert shapes
    assert X_sample.shape == X.shape
    assert y_sample.shape == y.shape
    return X_sample, y_sample

