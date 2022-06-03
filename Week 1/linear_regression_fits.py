import numpy as np
import matplotlib.pyplot as plt

# Set up and plot the dataset 
ys = np.array([1.1, 2.3, 2.9]) # N, 
X = np.array([[0.8], [1.9], [3.1]]) # N,1
plt.plot(X, ys, 'x', markersize=20, mew=2) # mew = marker edge width

# phi-functions to create various matrices of new features 
# from an original matrix of 1D inputs
def phi_linear(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])
def phi_quadratic(X):
    return np.hstack([np.ones((X.shape[0], 1)), X, X ** 2])
# fixed-width (2) RBF in 1D
def fw_rbf(X, cs):
    return np.exp(-(X - cs) ** 2 / 2.0)
def phi_rbf(X):
    return np.hstack([fw_rbf(X, c) for c in [1, 2, 3]])

# Fit linear regressor to the dataset (using least-squares) and plot the result
# @param phi_fn: function that takes Nx1 matrix of 1D inputs and returns NxM basis function values
# @param X: Nx1 matrix of 1D inputs
# @param ys: Nx1 matrix of outputs
def fit_and_plot(phi_fn, X, ys):
    phi = phi_fn(X) # N,K
    w_fit = np.linalg.lstsq(phi, ys, rcond=None)[0] # K,1
    X_grid = np.arange(0, 4, 0.01)[:, None] # N,1
    phi_grid = phi_fn(X_grid) # N,K
    f_grid = phi_grid @ w_fit # N,1

    plt.plot(X_grid, f_grid, linewidth=2)

# Fit and plot various basis functions
fit_and_plot(phi_linear, X, ys)
fit_and_plot(phi_quadratic, X, ys)
fit_and_plot(phi_rbf, X, ys)
plt.legend(["Data", "Linear", "Quadratic", "RBF"])
plt.xlabel('x')
plt.ylabel('f')
plt.show()

# Note: One can always fit N points exactly using N basis functions. 
#       As long as Phi is invertible, we can set w = Phi^-1 @ y.