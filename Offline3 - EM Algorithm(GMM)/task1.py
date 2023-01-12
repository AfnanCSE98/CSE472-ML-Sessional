# import the necessary packages
import numpy as np
import matplotlib.pyplot as plt
from EMGaussian import GaussianMixtureModel as GMM

# Load the data
X = np.loadtxt('data3D.txt')

# Fit the model with n_components varying from 1 to 10 and plot the log likelihood
log_likelihoods = []
n_components = []
for i in range(1, 11):
    gmm = GMM(n_components=i)
    gmm.fit(X)
    log_likelihood, n_component = gmm.get_logLikelihood_and_n_components()
    log_likelihoods.append(log_likelihood)
    n_components.append(n_component)

plt.plot(n_components, log_likelihoods)
plt.xlabel('Number of Components')
plt.ylabel('Log Likelihood')
plt.show()

