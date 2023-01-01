"""
Refer to: https://en.wikipedia.org/wiki/Precision_and_recall#Definition_(classification_context)
"""

import numpy as np

def accuracy(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    # calculate the number of correct predictions
    correct_predictions = np.sum(y_true == y_pred)

    # calculate the total number of predictions
    total_predictions = y_true.shape[0]

    # calculate the accuracy
    accuracy = correct_predictions / total_predictions

    return accuracy

    

def precision_score(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    # calculate the number of true positives
    true_positives = np.sum((y_true == 1) & (y_pred == 1))

    # calculate the number of predicted positives
    predicted_positives = np.sum(y_pred == 1)

    # calculate the precision
    precision = true_positives / predicted_positives

    return precision


def recall_score(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    # calculate the number of true positives
    true_positives = np.sum((y_true == 1) & (y_pred == 1))

    # calculate the number of actual positives
    actual_positives = np.sum(y_true == 1)

    # calculate the recall
    recall = true_positives / actual_positives

    return recall


def f1_score(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    # calculate the precision and recall
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    # calculate the f1 score
    f1 = 2 * (precision * recall) / (precision + recall)

    return f1