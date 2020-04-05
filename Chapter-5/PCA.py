import numpy as np
import matplotlib.pyplot as plt


num_samples = 200
cov = np.array([[1, 1], [1, 3.4]])
mean = np.array([0, 0])

""" plot data"""
x1, x2 = np.random.multivariate_normal(mean, cov, num_samples).T
X = np.vstack((x1, x2)).T
plt.plot(X[:, 0], X[:, 1], 'o')

cov_X = (1 / (num_samples - 1)) * np.dot(X.T, X)
eigen_vals, eigen_vecs = np.linalg.eig(cov_X)

""" eigen decomposition """
assert(np.allclose(np.dot(eigen_vecs * eigen_vals, eigen_vecs.T), cov_X))
# Swap columns since the highest eigenvalue is the second one
eigen_vecs = np.vstack((eigen_vecs[:, 1], eigen_vecs[:, 0]))
Z = np.dot(eigen_vecs.T, X.T).T
Z1 = np.vstack((Z[:, 0], np.ones(num_samples))).T
plt.plot(Z[:, 0], Z[:, 1], 'x')
plt.show()