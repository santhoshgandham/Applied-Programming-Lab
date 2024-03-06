import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

def f1(x):
    return x ** 2 + 3 * x + 8

def f2(x): 
    return 2*x + 3

def f3(x, y):
    return x**4 - 16*x**3 + 96*x**2 - 256*x + y**2 - 4*y + 262

def df3_dx(x, y):
    return 4*x**3 - 48*x**2 + 192*x - 256

def df3_dy(x, y):
    return 2*y - 4

def f4(x,y):
    return np.exp(-(x - y)**2) * np.sin(y)

def df4_dx(x, y):
    return -2 * np.exp(-(x - y)**2) * np.sin(y) * (x - y)

def df4_dy(x, y):
    return np.exp(-(x - y)**2) * np.cos(y) + 2 * np.exp(-(x - y)**2) * np.sin(y)*(x - y)

def f5(x):
    return np.cos(x)**4 - np.sin(x)**3 - 4*(np.sin(x)**2) + np.cos(x) + 1

def f6(x):
    return -(4*(np.cos(x)**3))*np.sin(x) - (3*(np.sin(x)**2))*np.cos(x) - 8*np.sin(x)*np.cos(x) - np.sin(x)

xlim1= [-5,5]
xlim3 =  [-10, 10]
ylim3 =  [-10, 10]
xlim4 = [-np.pi, np.pi]
xlim5 = [0, 2*np.pi]

def gradient_func_1D(function_1D, function_derivative, rnge):
    xbase = np.linspace(rnge[0], rnge[1], 100)
    ybase = function_1D(xbase) 

    bestx = 0.0005
    fig, ax = plt.subplots()
    ax.plot(xbase, ybase)
    xall, yall = [], []
    lnall, = ax.plot([], [], 'ro')
    lngood, = ax.plot([], [], 'go', markersize=10)

    # Learning rate
    lr = 0.1

    minimum_value_1D = function_1D(bestx)  # Initialize with a starting value

    def onestepderiv(s):
        nonlocal bestx, minimum_value_1D
        xall.append(bestx)
        y = function_1D(bestx)
        yall.append(y)
        x = bestx - function_derivative(bestx) * lr
        bestx = x
        y = function_1D(x)
        lngood.set_data(x, y)
        lnall.set_data(xall, yall)

        # Update the minimum_value if the new value is lower
        minimum_value_1D = min(minimum_value_1D, y)

    ani =FuncAnimation(fig, onestepderiv, frames=range(10), interval=50, repeat=False)
    plt.show()

    return minimum_value_1D

def gradient_func_2D(function_2D, x_derivative, y_derivative, bestx, besty ,xrange, yrange):

    bestcost=1000
    lr = 0.1
    # bestx = 3
    # besty = 3

    minimum_value_2D = function_2D(bestx, besty)
    min_x = bestx  # Initialize minimum x
    min_y = besty  # Initialize minimum y

    minimum_value_2D = function_2D(bestx, besty)

    def plot_3d_function():
        x = np.linspace(xrange[0], xrange[1], 100)
        y = np.linspace(yrange[0], yrange[1], 100)
        X, Y = np.meshgrid(x, y)
        Z = function_2D(X, Y)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='viridis')
        ax.scatter(min_x, min_y, function_2D(min_x, min_y), color='red', s=20, edgecolor='black')
        plt.show()

    for _ in range(bestcost):
        x = bestx - x_derivative(bestx, besty) * lr
        x = np.clip(x, xrange[0], xrange[1])  # Limit x within the range
        y = besty - y_derivative(bestx, besty) * lr
        y = np.clip(y, yrange[0], yrange[1])  # Limit y within the range
        bestx = x
        besty = y
        z = function_2D(x, y)
        minimum_value_2D = min(minimum_value_2D, z)

    plot_3d_function()  # Call the 3D plotting function within the optimization loop

    return minimum_value_2D

# Call the gradient_func function and print the minimum value
print("Minimum value of f1(x):", gradient_func_1D(f1, f2, xlim1))
print("Minimum value of f3(x):", gradient_func_2D(f3, df3_dx, df3_dy, 3.52, 0.91 ,xlim3, ylim3))
print("Minimum value of f4(x):", gradient_func_2D(f4, df4_dx, df4_dy, -1.5,-1.5, xlim4, xlim4))
print("Minimum value of f5(x):", gradient_func_1D(f5, f6, xlim5))