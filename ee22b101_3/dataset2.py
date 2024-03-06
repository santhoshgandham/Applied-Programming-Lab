import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Approximated time period by observing the original dataset
T1 = 2.5

# Function for regression method
def func_regression(t, k1, k2, k3, k4, T):
    return k1 * np.sin(2*np.pi*t/T) + k2 * np.sin(2*np.pi*(3*t)/T) + k3 * np.sin(2*np.pi*(5*t)/T) + k4

# Function for curve_fit method
def func_curvefit(t, K1, K2, K3, K4, T):
    return K1 * np.sin(2*np.pi*t/T) + K2 * np.sin(3*2*np.pi*t/(T)) + K3 * np.sin(5*2*np.pi*t/(T)) + K4

# Load dataset
data = np.loadtxt('dataset2.txt')
t = data[:, 0]
y = data[:, 1]

# Construct the M matrix for regression
M = np.column_stack((np.sin(2*np.pi*t/T1), np.sin(2*np.pi*(3*t)/T1), np.sin(2*np.pi*(5*t)/T1), np.ones_like(t)))

# Perform regression to estimate amplitudes
k1, k2, k3, k4 = np.linalg.lstsq(M, y, rcond=None)[0]

# Perform curve_fit method to estimate amplitudes and time period by providing suitable initial parameters
p, _ = curve_fit(func_curvefit, t, y, p0=[1, 1, 1, 0, 2.5])
K1, K2, K3, K4, T = p

# Print results for comparison
print(f'Amplitudes of sin(t), sin(3t), and sin(5t) from regression method: {k1}, {k2}, {k3}, {k4} with time period {T1}')
print(f'Amplitudes of sin(t), sin(3t), and sin(5t) from curve_fit method: {K1}, {K2}, {K3}, {K4} with time period {T}')

# Generate estimated y values for both methods
y_est_regression = func_regression(t, k1, k2, k3, k4, T1)
y_est_curvefit = func_curvefit(t, K1, K2, K3, K4, T)

# Plot the results for the regression method
plt.plot(t, y, label='Original Data')
plt.plot(t, y_est_regression, 'r--', label='Fitted Curve')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Regression Method')
plt.legend()
plt.savefig("dataset2_1.png")

# Plot the results for the curve_fit method
plt.plot(t, y, label='Original Data')
plt.plot(t, y_est_curvefit, 'r--', label='Fitted Curve')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Curve_fit Method')
plt.legend()
plt.savefig("dataset2_2.png")