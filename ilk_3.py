import matplotlib.pyplot as plt
import csv
import numpy as np

# Load data from the CSV file
headers = ["Car","Model","Engine cc","Car weight","CO2 emission"]
carData = []

with open('data.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    next(csvreader)
    for row in csvreader:
        carData.append(row)

data = np.array(carData)

# Extract the measurements: x1 = Engine cc, x2 = Car weight
temp = np.array(data[:, 2:4], dtype=int)

# Extract the dependent variable: y = CO2 emission
Y = np.array(data[:, 4], dtype=int)

# Create the independent variables matrix X
X = np.concatenate((np.ones((36, 1), dtype=int), temp), axis=1)

# Display the first 4 rows of X and Y
print("First 4 rows of X:")
print(X[:4])
print("\nDependent variable Y:")
print(Y[:4])

# Calculate the regression coefficients [β0, β1, β2]
varX = np.matmul(X.transpose(), X)
temp = np.linalg.inv(varX)
temp = np.matmul(temp, X.transpose())
beta = np.matmul(temp, Y)

print("\nRegression coefficients [β0, β1, β2]:")
print(beta)

# Create a 3D scatter plot of the data points (X1, X2, Y) and the regression plane
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 1], X[:, 2], Y, c='green', label='Data Points')

# Create a meshgrid for the regression plane
xx1 = np.outer(np.linspace(800, 2500, 32), np.ones(32))
xx2 = np.outer(np.ones(32), np.linspace(700, 1800, 32))
yy = beta[0] + beta[1] * xx1 + beta[2] * xx2

# Plot the regression plane
ax.plot_surface(xx1, xx2, yy, alpha=0.5, color='blue', label='Regression Plane')

# Set labels for the axes
ax.set_xlabel('Engine cc', fontweight='bold')
ax.set_ylabel('Car weight', fontweight='bold')
ax.set_zlabel('CO2 emission', fontweight='bold')

# Add a legend
ax.legend()

# Show the plot
plt.show()
