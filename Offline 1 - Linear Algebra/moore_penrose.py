import numpy as np

#Take the dimensions of matrix n, m as input
n = int(input("Enter n: "))
m = int(input("Enter m: "))

#Produce a random n x m matrix A. For the purpose of demonstrating, every cell of A must be an integer
A = np.random.randint(1, 100, (n, m))

#Print A
print("Matrix = \n", A)

#Perform Singular Value Decomposition
U, D, V = np.linalg.svd(A)

print("Singular Values: \n", D)
print("Left Singular Vectors: \n", U)
print("Right Singular Vectors: \n", V)

#convert D to a diagonal matrix,then reciprocal of its nonzero elements and then take transpose
D = np.diag(D)
Z = np.zeros((m, n))
np.fill_diagonal(Z, 1/D.diagonal())

#Calculate the Moore-Penrose Pseudoinverse using NumPy’s builtin function
A_pseudoinverse_1 = np.linalg.pinv(A)

#calculate pseudoinverse using U,D,V
A_pseudoinverse_2 = V.T.dot(Z).dot(U.T)

#Check if these two inverses are equal
print("Are the pseudoinverses the same? ", np.allclose(A_pseudoinverse_1, A_pseudoinverse_2))



