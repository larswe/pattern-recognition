import numpy as np
import matplotlib.pyplot as plt

# Note: Using quadratic basis functions with different centers is not very effective 
#       because any linear combination of the functions will simply be a single quadratic function. 

# Random weights
K = 10
ws = np.random.randn(K)
centers = 10 * np.random.randn(K)

# Quadratic basis function
X = np.linspace(-30, 30, 100).reshape(-1, 1)
# Matrix with columns [(X - c)^2 for c in centers]
X_quad = np.hstack([(X - c) ** 2 for c in centers])

# Plot function 
ys = X_quad @ ws
plt.title("A linear combination of %d quadratic functions" % K)
plt.plot(X, ys, '-')
plt.show()