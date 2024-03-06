# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
data = np.loadtxt('dataset1.txt')

# Extract x and y values from the dataset
x = data[:, 0]
y = data[:, 1]

# Construct the M matrix for least squares
M = np.column_stack((x, np.ones_like(x)))

# Perform least squares regression to estimate the slope and intercept
m, c = np.linalg.lstsq(M, y, rcond=None)[0]

# Print the optimal values of the slope and intercept
print(f'Optimal value of the slope: {m}')
print(f'Optimal value of the intercept: {c}')

# Plot the original data points
plt.plot(x, y, label='Data Points')

# Plot the estimated line over the original data
plt.plot(x, m*x + c, 'r--', label='Estimated')

# Plot error bars for every 25 data points
plt.errorbar(x[::25], y[::25], np.std(y-(m*x + c)), fmt='ro', label='Error Bars')

# Set labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Linear Regression using Least Squares Method')

# Add legend
plt.legend()

#save the plot
plt.savefig("dataset1.png")