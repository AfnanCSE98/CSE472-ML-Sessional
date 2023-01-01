import numpy as np

class PCA:
    def __init__(self, n_new_features):
        self.n_new_features = n_new_features

    def fit_transform(self, X):
         # Center the data
        X = X - np.mean(X, axis=0)

        # Compute the covariance matrix
        cov_matrix = np.cov(X, rowvar=False)

        # Compute the eigenvalues and eigenvectors of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # Sort the eigenvalues and eigenvectors in decreasing order
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:,idx]

        # Select the top n_components eigenvectors
        self.components_ = eigenvectors[:, :self.n_new_features]

        # Project the data onto the principal components
        X_reduced = np.dot(X, self.components_)

        return X_reduced


