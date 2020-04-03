import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
""" Without regularization """

# 1 Dimensional
num_samples = 10
num_features = 1
theta = np.ones((1,1))* 3.14
bias = np.ones((1,1)) * 4.2
noise_sigma = 1
X_train = np.arange(num_samples)
X_train = X_train[:, None]
# X_train = np.random.randn(num_samples, num_features)
X_train_bias = np.hstack((X_train, np.ones((num_samples, 1))))
y_train = np.matmul(X_train, theta) + bias
y_train_noise = y_train + noise_sigma * np.random.randn(*y_train.shape)
plt.scatter(X_train[:, 0], y_train_noise, facecolors="none", edgecolors="b")
plt.plot(y_train, color="red")
plt.grid()
plt.show()
w = np.matmul(np.matmul(inv(np.matmul(X_train_bias.T, X_train_bias)), X_train_bias.T), y_train_noise)
theta_hat = w[:-1]
bias_hat = w[-1]
print(f"{num_features} Dimensions")
print(f"Theta:\n{theta} \nTheta_hat:\n{theta_hat}")
print(f"\nBias:\n{bias} \nBias_hat:\n{bias_hat}")
print("\n*************")

# n Dimensional
num_samples = 10
num_features = 3
theta = np.random.randn(3, 1)
bias = np.random.randn(1,1)
X_train = np.random.randn(num_samples, num_features)
X_train_bias = np.hstack((X_train, np.ones((num_samples, 1))))
y_train = np.matmul(X_train, theta) + bias

w = np.matmul(np.matmul(inv(np.matmul(X_train_bias.T, X_train_bias)), X_train_bias.T), y_train)
theta_hat = w[:-1]
bias_hat = w[-1]
print(f"\n{num_features} Dimensions")
print(f"Theta:\n{theta} \nTheta_hat:\n{theta_hat}")
print(f"\nBias:\n{bias} \nBias_hat:\n{bias_hat}")
