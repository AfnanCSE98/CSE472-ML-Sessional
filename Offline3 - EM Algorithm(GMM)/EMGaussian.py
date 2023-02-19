import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

class GaussianMixtureModel:
    def __init__(self, n_components, max_iter=100, tol=0.2):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.weights_ = None
        self.means_ = None
        self.covariances_ = None

    def fit(self, X , plot = False):
        n_samples, n_features = X.shape

        # Initialize the weights, means, and covariances
        weights = np.ones(self.n_components) / self.n_components
        means = np.random.rand(self.n_components, n_features)
        # take 10% of data randomly and initialize covariance matrix
        covariances = np.array([np.cov(X[np.random.choice(n_samples, int(n_samples/10), replace=False)], rowvar=False) for _ in range(self.n_components)])
        # perform error checking
        if n_samples < self.n_components:
            raise ValueError('The number of samples must be greater than the number of components')

        # Initialize the responsibility matrix
        resp = np.zeros((n_samples, self.n_components))

        # Initialize the log likelihood
        log_likelihood = None

        print("---------------- n_components : " , self.n_components, " --------------------")

        # Run the EM algorithm
        for i in range(self.max_iter):
            # E-step: calculate the responsibilities
            resp = self._e_step(X, weights, means, covariances)

            # M-step: update the weights, means, and covariances
           
            weights, means, covariances = self._m_step(X, resp)
            # Calculate the log likelihood
            log_likelihood_new = self._calculate_log_likelihood(X, weights, means, covariances)

            # print after checkpoints
            if i % 10 == 0:
                print('Iteration: {}, Log-Likelihood: {}'.format(i+10, log_likelihood_new))

            # Check for convergence
            if log_likelihood is not None and abs(log_likelihood - log_likelihood_new) < self.tol:
                break
            log_likelihood = log_likelihood_new

            # # plot the data points and gaussian distributions in a 2D plot
            if plot:
                
                self._plot(X, weights, means, covariances , i+1)


        self.weights_ = weights
        self.means_ = means
        self.covariances_ = covariances
        self.log_likelihood_ = log_likelihood
    

    def _e_step(self, X, weights, means, covariances):
        """
        The E-step, or "Expectation step", involves computing the "responsibility" of each component for each data point.
        This is done by evaluating the probability of each data point under each component, given the current model parameters, 
        and normalizing these probabilities to sum to 1 for each data point.
        The resulting matrix of probabilities is called the "responsibility matrix".
        <<<<<<<<<<"responsibility matrix" has dimenson (n_samples, n_components)>>>>>>>>>>>>>
        """
        n_samples, _ = X.shape
        resp = np.zeros((n_samples, self.n_components))

        # Calculate the responsibility matrix without numpy methods
        for i in range(self.n_components):
            resp[: , i] = weights[i] * self._multivariate_normal(X, means[i], covariances[i])
    
        # replace any entry of resp if that is zero with 0.0001
        for i in range(n_samples):
            for j in range(self.n_components):
                if resp[i,j] == 0:
                    resp[i,j] = 0.0001
        # Normalize the resp matrix
        resp /= resp.sum(axis=1, keepdims=True)

        return resp 

    def _m_step(self, X, resp):
        """
        The M-step, or "Maximization step", involves updating the model parameters
        (i.e., the weights, means, and covariances of the components) using the responsibility matrix 
        computed in the E-step. This is done by maximizing the log-likelihood of the data given the responsibilities.
        """
        n_samples, n_features = X.shape
        weights = np.zeros(self.n_components)
        means = np.zeros((self.n_components, n_features))
        covariances = np.array([np.eye(n_features) for _ in range(self.n_components)]) 

        for i in range(self.n_components):
            # Update the weights
            weights[i] = resp[:, i].sum() / n_samples
            # Update the means and covariances without using numpy methods
            m_sum = np.zeros(n_features)
            for j in range(n_samples):
                m_sum += resp[j, i] * X[j]

            # check if m_sum or resp[:,i].sum() is nan
          
            means[i] = m_sum / resp[:, i].sum()
            cv_sum = np.zeros((n_features, n_features))
            for j in range(n_samples):
                cv_sum += resp[j, i] * np.outer(X[j] - means[i], X[j] - means[i])
            covariances[i] = cv_sum / resp[:, i].sum()

        #if means or covariances contain nan
        if np.isnan(means).any() or np.isnan(covariances).any():
            print('Means or covariances contain nan')
            print('means: {}, covariances: {}'.format(means, covariances))
            raise ValueError('Means or covariances contain nan')
        return weights, means, covariances

    def _calculate_log_likelihood(self, X, weights, means, covariances):
        """
        Calculate the log likelihood of the data given the current model parameters.
        """
        log_likelihood = 0
    
        for i in range(self.n_components):
           
            log_likelihood += weights[i] * self._multivariate_normal(X, means[i], covariances[i])

        return np.sum(np.log(log_likelihood))

    def _multivariate_normal(self, X, mean, covariance):
        # n_samples, n_features = X.shape
        # det = np.linalg.det(covariance)
        # inv = np.linalg.inv(covariance)

        # norm_const = 1.0 / (np.power((2 * np.pi), float(n_features) / 2) * np.power(det, 1.0 / 2))

        # return norm_const * np.exp(-0.5 * np.einsum('ij, ij -> i', X - mean, np.dot(X - mean, inv)))
        
        return multivariate_normal.pdf(X , mean , covariance , allow_singular=True)

    def predict(self, X , weights, means, covariances):
        
        resp = self._e_step(X, weights, means, covariances)

        # find the indices of the maximum values along the second axis (i.e., the column axis) of the responsibility matrix
        # [ex : np.array([[1, 2], [5, 6], [7, 9]]) => (3 columns , 2 rows) => np.argmax(resp, axis=1) => [1, 1, 1]
        return np.argmax(resp, axis=1)

    def get_logLikelihood_and_n_components(self):
        return self.log_likelihood_, self.n_components

    def _plot(self , X , weights , means , covariances , iteration):
        color_ara = self.predict(X , weights , means , covariances)
        plt.scatter(X[:, 0], X[:, 1], c = color_ara)
        # set heading of the plot as iteration number
        plt.title('Iteration: {}'.format(iteration))
        plt.show()