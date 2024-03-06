import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Constants
h = 6.63e-34  # Planck's constant
c = 3e8       # Speed of light
k = 1.38e-23  # Boltzmann constant

# Load dataset
data = np.loadtxt('dataset3.txt')
f = data[:, 0]
B = data[:, 1]

# Define the function to fit
def func(f, T):
    """
    Planck's Law for blackbody radiation.

    Parameters:
    - f: Frequency
    - T: Temperature

    Returns:
    - The spectral radiance of a blackbody at temperature T and frequency f.
    """
    return (2 * h * (f**3) / (c**2)) / (np.exp((h * f) / (k * T)) - 1)

# Perform curve fitting to estimate the temperature
(T), _ = curve_fit(func, f, B, p0=[4000])

# Print the optimal value of T
print(f'Optimal value of T: {T}')

# Generate estimated values using the fitted parameters
B_est = func(f, T)

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
plt.savefig("dataset3_1.png")