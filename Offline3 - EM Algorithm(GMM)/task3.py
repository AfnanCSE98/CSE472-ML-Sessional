# import the necessary packages
import numpy as np
import matplotlib.pyplot as plt
from EMGaussian import GaussianMixtureModel as GMM
from pca import PCA

# Load the data
X = np.loadtxt('data3D.txt')
print(X.shape)

pca = PCA(n_new_features=2)
X = pca.fit_transform(X)
print(X.shape)

n_components = 4
# Fit the model with n_components and plot= True
gmm = GMM(n_components=n_components)

gmm.fit(X, plot=True)
