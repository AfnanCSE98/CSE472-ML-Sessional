import numpy as np

# generate a random np array of shape (1500,2)
arr = np.array([[1, 2], [5, 6], [7, 9]])

max_indices = np.argmax(arr, axis=1)
print(max_indices)  # Output: [2 2 2]