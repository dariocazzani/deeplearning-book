import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Short sample app')
parser.add_argument('--plot', action="store_true", default=False)
args = parser.parse_args()

""" Without regularization """
# 1 Dimensional
num_samples = 100
num_features = 1
theta = np.ones((1,1))* 3.14
offset = np.ones((1,1)) * 4.2
noise_sigma = 3
X_train = np.arange(-num_samples, num_samples+1)
num_samples = len(X_train)
X_train = X_train[:, None]
# X_train = np.random.randn(num_samples, num_features)
X_train_offset = np.hstack((X_train, np.ones((num_samples, 1))))
y_train = np.matmul(X_train, theta) + offset
y_train_noise = y_train + noise_sigma * np.random.randn(*y_train.shape)

if args.plot:
    plt.scatter(X_train[:, 0], y_train_noise, facecolors="none", edgecolors="b")
    plt.plot(X_train, y_train, color="red")
    plt.grid()
    plt.show()

w = np.matmul(np.matmul(inv(np.matmul(X_train_offset.T, X_train_offset)), X_train_offset.T), y_train_noise)
theta_hat = w[:-1]
offset_hat = w[-1]
print(f"{num_features} Dimensions")
print(f"Theta:\n{theta} \nTheta_hat:\n{theta_hat}")
print(f"\nOffset:\n{offset} \nOffset_hat:\n{offset_hat}")

# Compute an unbiased estimate of the variance (of noise) as the MSE
y_pred = np.matmul(X_train, theta_hat) + offset_hat
noise_variance_hat = np.sum((y_pred-y_train)**2) / (num_samples-2)
print(f"\nOriginal variance: {noise_sigma**2}")
print(f"Estimated variance: {noise_variance_hat}")

print("\n*************")
# n Dimensional
num_samples = 10
num_features = 3
theta = np.random.randn(3, 1)
offset = np.random.randn(1,1)
X_train = np.random.randn(num_samples, num_features)
X_train_offset = np.hstack((X_train, np.ones((num_samples, 1))))
y_train = np.matmul(X_train, theta) + offset

w = np.matmul(np.matmul(inv(np.matmul(X_train_offset.T, X_train_offset)), X_train_offset.T), y_train)
theta_hat = w[:-1]
offset_hat = w[-1]
print(f"\n{num_features} Dimensions")
print(f"Theta:\n{theta} \nTheta_hat:\n{theta_hat}")
print(f"\nOffset:\n{offset} \nOffset_hat:\n{offset_hat}")
