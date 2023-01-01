#import numpy
import numpy as np

#Take the dimensions of matrix n as input
n = int(input("Enter the dimensions of the matrix: "))

#Produce a random n x n invertible symmetric matrix A. For the purpose of demonstrating, every cell of A will be an integer
A = np.random.randint(0, 100, (n, n))
A = A + A.T

#make sure A is invertible and symmetric
while np.linalg.det(A) == 0 or not np.allclose(A, A.T):
    A = np.random.randint(0, 100, (n, n))

#print A
print("Matrix = ", A)

#Perform Eigen Decomposition 
eigenvalues, eigenvectors = np.linalg.eig(A)

#Print the eigenvalues and eigenvectors
print("Eigenvalues: ", eigenvalues)
print("Eigenvectors: ", eigenvectors)

#Reconstruct A from eigenvalue and eigenvectors
A_reconstructed = eigenvectors.dot(np.diag(eigenvalues)).dot(np.linalg.inv(eigenvectors))

#Print the reconstructed matrix
print("Reconstructed Matrix: ", A_reconstructed)

#Check if the reconstructed matrix is the same as the original matrix
print("Are the matrices the same? ", np.allclose(A, A_reconstructed))



