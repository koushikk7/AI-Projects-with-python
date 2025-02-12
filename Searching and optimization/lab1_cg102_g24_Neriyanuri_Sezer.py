import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def function(x, y):
    """Defines the objective function."""
    return 2 * np.sin(x) + 3 * np.cos(y)

def gradient(x, y):
    """Computes the gradient of the objective function."""
    df_dx = 2 * np.cos(x)
    df_dy = -3 * np.sin(y)
    return np.array([df_dx, df_dy])

def hessian(x, y):
    """Computes the Hessian matrix of the objective function."""
    d2f_dxdx = -2 * np.sin(x)
    d2f_dydy = -3 * np.cos(y)
    return np.array([[d2f_dxdx, 0], [0, d2f_dydy]])

def newton_method(initial_guess, learning_rate=1.0, tol=1e-6, max_iter=1000):
    """Performs Newton's method for function minimization with a learning rate."""
    x = np.array(initial_guess)
    for i in range(max_iter):
        grad = gradient(x[0], x[1])
        hess_inv = np.linalg.inv(hessian(x[0], x[1]))
        alpha_x = learning_rate * np.dot(hess_inv, grad)
        x -= alpha_x
        if np.linalg.norm(alpha_x) < tol:
            return x, i + 1
    return x, max_iter

def visualize():
    """Visualizes the objective function surface."""
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = function(X, Y)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

visualize()

initial_guess_1 = [2.0, 2.0]
minimum_1, iterations_1 = newton_method(initial_guess_1, learning_rate=1.0)
print(f"Minimum approximation with initial guess {initial_guess_1}: {minimum_1}, Iterations: {iterations_1}")
