import matplotlib.pyplot as plt
import csv
import numpy as np
from scipy import linalg


headers = ["Car","Model","Engine cc","Car weight","CO2 emission"]
carData = []   # will list of 36 car information lists

with open('data.csv', 'r') as csvfile:  # open data.csv for reading 
    csvreader = csv.reader(csvfile)
    next(csvreader)                     # skip data headers
    for row in csvreader:               # extract data rows one by one
        carData.append(row)             # carData[i] = i-th car info
data = np.array(carData)                # convert carData list into array 

# take the measurements  x1 = Engine cc, x2 = Car weight
temp = np.array(data[:,2:4],dtype=int)
# take the measurements  y  = CO2 emission (dependent variable)
Y = np.array(data[:,4],dtype=int)
# X[i] = [1, x1, x2]
X = np.concatenate((np.ones((36,1),dtype=int), temp),axis=1)

varX = np.matmul(X.transpose(),X)       # sxx = X^T * X
temp = np.linalg.inv(varX)              # (X^T * X)^(-1)
temp = np.matmul(temp,X.transpose())    # (X^T * X)^(-1) * X^T
beta = np.matmul(temp,Y)                # (X^T * X)^(-1) * X^T * Y

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(X[:,1],X[:,2],Y, 'green')
ax.set_xlabel('Engine cc', fontweight ='bold') 
ax.set_ylabel('Car weight', fontweight ='bold') 
ax.set_zlabel('CO2 emission', fontweight ='bold')

xx1 = np.outer(np.linspace(800, 2500, 32), np.ones(32))
xx2 = np.outer(np.ones(32)               , np.linspace(700, 1800, 32))
yy = ( beta[0] + beta[1]*xx1 + beta[2]*xx2 )
ax.plot_surface(xx1, xx2, yy, alpha=0.25)

plt.show()



