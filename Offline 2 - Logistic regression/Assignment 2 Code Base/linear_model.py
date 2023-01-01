import numpy as np

class LogisticRegression:
    def __init__(self, params):
        """
        figure out necessary params to take as input
        :param params:
        """
        self.learning_rate = params["learning_rate"]
        self.num_iterations = params["num_iterations"]
        self.fit_intercept = params["fit_intercept"]
        

    def fit(self, X, y):
        """
        :param X:
        :param y:
        :return: self
        """
        assert X.shape[0] == y.shape[0]
        assert len(X.shape) == 2
        # Add a column of ones to X if fit_intercept is True
        if self.fit_intercept:
            X = np.c_[np.ones((X.shape[0], 1)), X]
        
        # Initialize the weights to zeros
        self.theta = np.zeros(X.shape[1])
        
        # Loop over the number of iterations
        for i in range(self.num_iterations):
            # Compute the predicted probabilities for X using sigmoid function
            probabilities = self._sigmoid(X)
            
            # Compute the error term for each sample
            error = y - probabilities
            
            # Compute the gradient of the loss function with respect to theta
            gradient = np.dot(X.T, error)
            
            # Update the weights using the gradient descent update rule
            self.theta += self.learning_rate * gradient


    def predict(self, X):
        """
        function for predicting labels of for all datapoint in X
        :param X:
        :return:
        """
        # Add a column of ones to X if fit_intercept is True
        if self.fit_intercept:
            X = np.c_[np.ones((X.shape[0], 1)), X]
        
        # Compute the predicted probabilities for X using sigmoid function
        probabilities = self._sigmoid(X)
        
        # Return the predicted class labels as 0 or 1
        return (probabilities > 0.5).astype(int)

    def _sigmoid(self, X):
        """
        function for computing sigmoid
        :param X:
        :return:
        """
        return 1 / (1 + np.exp(-np.dot(X, self.theta)))
