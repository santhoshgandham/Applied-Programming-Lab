import numpy as np
import matplotlib.pyplot as plt

# Load data from CSV file
data = np.loadtxt("Admission_Predict_Ver1.1.csv", delimiter=",", skiprows=1)

# Feature scaling (normalising)
GRE = data[:, 1] / 340
TOEFL = data[:, 2] / 120
UR = data[:, 3] / 100
SOP = data[:, 4] / 5
LOR = data[:, 5] / 5
CGPA = data[:, 6] / 10
Research = data[:, 7]
COA = data[:, 8]

# Define the admission probability function
def admission_probability_func(GRE, GRE_p, TOEFL, TOEFL_p, UR, UR_p, SOP, SOP_p, LOR, LOR_p, CGPA, CGPA_p, Research, Research_p, constant):
    return GRE * GRE_p + TOEFL * TOEFL_p - UR * UR_p + SOP * SOP_p + LOR * LOR_p + CGPA * CGPA_p + Research * Research_p + constant

# Define the linear regression model matrix
M = np.column_stack((GRE, TOEFL, -UR, SOP, LOR, CGPA, Research, np.ones_like(COA)))

# Perform linear regression to find model coefficients
GRE_p, TOEFL_p, UR_p, SOP_p, LOR_p, CGPA_p, Research_p, constant = np.linalg.lstsq(M, COA, rcond=None)[0]

# Print model coefficients
print("GRE coefficient     :", GRE_p)
print("TOEFL coefficient   :", TOEFL_p)
print("SOP coefficient     :", SOP_p)
print("LOR coefficient     :", LOR_p)
print("CGPA coefficient    :", CGPA_p)
print("Research coefficient:", Research_p)

# Calculate admission probabilities
prob_est = admission_probability_func(GRE, GRE_p, TOEFL, TOEFL_p, UR, UR_p, SOP, SOP_p, LOR, LOR_p, CGPA, CGPA_p, Research, Research_p, constant)

# Calculate percentage accuracy
b = float()
for i in range(len(data)):
    b += ((abs(COA[i] - prob_est[i])/COA[i]) * 100 / len(data))
c = 100 - b
print(f"\n%accuracy= {c}")

# Plotting
plt.plot(COA, prob_est, 'r.',label=(f"{c}% accurate prob fit"))
plt.xlabel('Chance of Admit')
plt.ylabel('Estimated Probability')
plt.plot(COA, COA, 'b-',label= '100% accurate prob fit')
plt.legend()
plt.savefig("APL-aasgn4.png")  # Save the plot as an image
