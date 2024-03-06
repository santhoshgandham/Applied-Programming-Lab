import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Load dataset
data = np.loadtxt('dataset3.txt')
f = data[:, 0]
B = data[:, 1]

# Define the function to fit
def func(f, h, c, k, T):
    """
    Planck's Law for blackbody radiation.

    Parameters:
    - f: Frequency
    - h: Planck's constant
    - c: Speed of light
    - k: Boltzmann constant
    - T: Temperature

    Returns:
    - The spectral radiance of a blackbody at temperature T and frequency f.
    """
    return (2 * h * (f ** 3) / (c ** 2)) / (np.exp((h * f) / (k * T)) - 1)

# Perform curve fitting to estimate the parameters
h, c, k, T = curve_fit(func, f, B, p0=[2.881433427421395e-33, 626314803.1705209, 5.791777742696577e-23, 4000])[0]

# Print the optimal values
print(f'Optimal value of h: {h}')
print(f'Optimal value of c: {c}')
print(f'Optimal value of k: {k}')
print(f'Optimal value of T: {T}')

# Generate estimated values using the fitted parameters
B_est = func(f, h, c, k, T)

# Plot the original data and the fitted curve
plt.plot(f, B, label='Original Data')
plt.plot(f, B_est, 'r--', label='Fitted Curve')

# Set labels and title
plt.xlabel('Frequency')
plt.ylabel('Spectral Radiance')
plt.title('Blackbody Radiation Curve Fitting')

# Add legend
plt.legend()

#save the plot
plt.savefig("dataset3_2.png")
