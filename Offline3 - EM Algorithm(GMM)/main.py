import numpy as np
import matplotlib.pyplot as plt

class GaussianMixtureModel:
    def __init__(self, n_components, max_iter=100, tol=1e-3):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        n_samples, n_features = X.shape

        # Initialize the weights, means, and covariances
        weights = np.ones(self.n_components) / self.n_components
        means = np.random.rand(self.n_components, n_features)
        covariances = np.array([np.eye(n_features) for _ in range(self.n_components)])

        # perform error checking
        if n_samples < self.n_components:
            raise ValueError('The number of samples must be greater than the number of components')

        # Initialize the responsibility matrix
        resp = np.zeros((n_samples, self.n_components))

        # Initialize the log likelihood
        log_likelihood = None

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
                print('Iteration: {}, Log-Likelihood: {}'.format(i, log_likelihood_new))

            # Check for convergence
            if log_likelihood is not None and abs(log_likelihood - log_likelihood_new) < self.tol:
                break
            log_likelihood = log_likelihood_new

        self.weights_ = weights
        self.means_ = means
        self.covariances_ = covariances
        self.log_likelihood_ = log_likelihood

    def _e_step(self, X, weights, means, covariances):
        n_samples, _ = X.shape
        resp = np.zeros((n_samples, self.n_components))
        for i in range(self.n_components):
            resp[:, i] = weights[i] * self._multivariate_normal(X, means[i], covariances[i])
        resp /= resp.sum(axis=1)[:, np.newaxis]
        return resp

    def _m_step(self, X, resp):
        n_samples, n_features = X.shape
        weights = resp.sum(axis=0) / n_samples
        means = np.zeros((self.n_components, n_features))
        covariances = np.zeros((self.n_components, n_features, n_features))
        for i in range(self.n_components):
            means[i] = (resp[:, i][:, np.newaxis] * X).sum(axis=0) / resp[:, i].sum()
            covariances[i] = (resp[:, i][:, np.newaxis] * ( X - means[i] )).T.dot(X - means[i]) / resp[:, i].sum()
        return weights, means, covariances

    def _calculate_log_likelihood(self, X, weights, means, covariances):
        log_likelihood = 0
        for i in range(self.n_components):
            log_likelihood += weights[i] * self._multivariate_normal(X, means[i], covariances[i])
        return np.log(log_likelihood).sum()

    def _multivariate_normal(self, X, mean, covariance):
        n_samples, n_features = X.shape
        det = np.linalg.det(covariance)
        inv = np.linalg.inv(covariance)
        norm_const = 1.0 / (np.power((2 * np.pi), float(n_features) / 2) * np.power(det, 1.0 / 2))
        return norm_const * np.exp(-0.5 * np.einsum('ij, ij -> i', X - mean, np.dot(X - mean, inv)))

    def predict(self, X):
        resp = self._e_step(X, self.weights_, self.means_, self.covariances_)
        return np.argmax(resp, axis=1)

if __name__ == '__main__':
    # Load the data
    X = np.loadtxt('data2D.txt')

    # Fit the model
    gmm = GaussianMixtureModel(n_components=2)
    gmm.fit(X)

    # Plot the data

    plt.scatter(X[:, 0], X[:, 1], c=gmm.predict(X))
    plt.show()
