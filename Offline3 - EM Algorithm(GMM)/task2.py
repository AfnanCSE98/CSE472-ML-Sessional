# import the necessary packages
import numpy as np
import matplotlib.pyplot as plt
from EMGaussian import GaussianMixtureModel as GMM

# Load the data
X = np.loadtxt('data2D.txt')

n_components = 3

# Fit the model with n_components and plot= True
gmm = GMM(n_components=n_components)

gmm.fit(X, plot=True)


