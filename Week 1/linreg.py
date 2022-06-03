import numpy as np 
import matplotlib.pyplot as plt

print("-" * 50)
print("Linear Regression without bias term")
print("-" * 50)

# Dimensions
N, D = 7, 3

# Data
X = np.random.randn(N, D)

# Weights
ww = np.random.randn(D)
print("Real weights:", ww)

# Targets without noise
yy = X @ ww

# Fit the model
w_fit = np.linalg.lstsq(X, yy, rcond=None)[0]
print("Estimate without noise:", w_fit)

# Targets with noise
yy = yy + np.random.randn(N) * 0.1

# Fit the model
w_fit = np.linalg.lstsq(X, yy, rcond=None)[0]
print("Estimate with noise:", w_fit)

print("-" * 50)
print("Note that yy has shape", yy.shape)
print("It is important that yy is a vector, not a matrix.")

###############################################################################

print("-" * 50)
print("Linear Regression with bias term")
print("-" * 50)

# Bias term
bias = 3 + np.random.randn(1)
b = bias + np.random.randn(N) * 0.1

# Targets with noise and bias
yy = yy + b

# Extend X with a column of ones
X_bias = np.hstack([np.ones((N, 1)), X])

# Fit the model
w_fit = np.linalg.lstsq(X_bias, yy, rcond=None)[0]

print("Real bias and weights:", np.hstack([bias, ww]))
print("Estimate:", w_fit)


###############################################################################

print("-" * 50)
print("Linear Regression with polynomial features")
print("-" * 50)

# Univariate data
N = 50
X = np.random.randn(N, 1)

# Use polynomial basis function
degree = 3
X_poly = np.hstack([X ** i for i in range(degree + 1)])

# Weights
ww = np.random.randn(degree + 1)

# Targets with noise
yy = X_poly @ ww + np.random.randn(N) * 1.5

# Fit the model
w_fit = np.linalg.lstsq(X_poly, yy, rcond=None)[0]

print("Real weights:", ww)
print("Estimate:", w_fit)

# Plot fit
plt.plot(X, yy, 'o')
X_plot = np.linspace(-3, 3, 100).reshape(-1, 1) # Linspace as matrix
X_plot_poly = np.hstack([X_plot ** i for i in range(degree + 1)])
plt.plot(X_plot, X_plot_poly @ w_fit, '-')
plt.title("Polynomial fit of degree %d" % degree)
plt.show()

print("-" * 50)