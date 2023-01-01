from data_handler import bagging_sampler
import numpy as np
from collections import Counter

class BaggingClassifier:
    def __init__(self, base_estimator, n_estimator):
        """
        :param base_estimator:
        :param n_estimator:
        :return:
        """
        self.base_estimator = base_estimator
        self.n_estimator = n_estimator


    def fit(self, X, y):
        """
        :param X:
        :param y:
        :return: self
        """
        assert X.shape[0] == y.shape[0]
        assert len(X.shape) == 2
        
        self.estimators_ = []
        for i in range(self.n_estimator):
            estimator = self.base_estimator
            X_sample, y_sample = bagging_sampler(X, y)
            estimator.fit(X_sample, y_sample)
            self.estimators_.append(estimator)

    def predict(self, X):
        """
        function for predicting labels of for all datapoint in X
        apply majority voting
        :param X:
        :return:
        """
        #predict labels for each datapoint in X for each estimator and apply majority voting
        predictions = []
        y_pred = []
        for estimator in self.estimators_:
            y_pred.append(estimator.predict(X))
        
        for i in range(X.shape[0]):
            _ = [item[i] for item in y_pred]
            most_common = Counter(_).most_common()[0][0] # get the most frequent value in y_pred  
            predictions.append(most_common)

        return np.array(predictions)