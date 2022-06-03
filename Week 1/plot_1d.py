import numpy as np 
import matplotlib.pyplot as plt

# Plot cosine function
grid_size = 0.1
x_grid = np.arange(-10, 10, grid_size)
f_vals = np.cos(x_grid)
plt.plot(x_grid, f_vals, "b-")
plt.plot(x_grid, f_vals, "r.")
plt.show()

###############################################################################

# RBF
# @param xs: input vector
# @param c: center
# @param h: bandwidth
# @return: output vector
def rbf_1d(xs, c, h):
    return np.exp(-((xs - c) ** 2) / (h ** 2))

plt.clf() # clear figure

grid_size = 0.01
x_grid = np.arange(-10, 10, grid_size)
plt.plot(x_grid, rbf_1d(x_grid, c=5, h=1), "-b")
plt.plot(x_grid, rbf_1d(x_grid, c=-2, h=2), "-r")
plt.show()

###############################################################################

# 1D Logistic-sigmoid function
# @param xs: input vector
# @param v: weight
# @param b: bias
# @return: output vector
def logistic_sigmoid_1d(xs, v, b):
    return 1 / (1 + np.exp(-v * xs - b))

plt.clf()

vs = np.array([-1, 0.5, 1])
bs = np.array([-3, 3])

x_grid = np.arange(-10, 10, grid_size)
# Plot sigmoid for each v and b, with legend and color
for v in vs:
    for b in bs:
        plt.plot(x_grid, logistic_sigmoid_1d(x_grid, v, b), "--", label="v={}, b={}".format(v, b))
plt.title("Logistic-sigmoid functions")
plt.legend()
plt.show()

# Note: 
# The logistic-sigmoid function is equal to 0.5 if v * x + b = 0. 
# Its derivative is equal to v * exp(v * x + b) / (1 + exp(v * x + b))^2 and is steepest if v * x + b = 0.
# The steepest slope is thus equal to v / 4.
# Hence, v determines the steepness of the curve, whereafter b determines the position of the steepest slope.