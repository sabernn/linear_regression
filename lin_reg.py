
######## TITLE ########
# CSC 7333 Programming Assignment
# Instructor: Dr. Jianhua Chen
# Developed by: Saber Nemati
# email: mnemat2@lsu.edu


# Importing packages
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt


# Declaring the loss function
def loss_func(inp,out,theta):
    m=len(inp)
    J=0
    for i in range(m):
        J+=(theta[0]+theta[1]*inp[i]-out[i])**2

    return J/2/m

# Declaring the update term function
def loss_func_der(inp,out,theta):
    m=len(inp)
    dJ=0
    for i in range(m):
        dJ+=(-theta[0]-theta[1]*inp[i]+out[i])

    return dJ/m


########     INPUT DATA    ########
# Number of iterations
n=15
# Learning rate
alpha=0.01
######## END OF INPUT DATA ########



# Initializing input and output vectors
x=[]
y=[]
# Importing data
with open('data//KCSmall2.csv', newline='') as f:
    reader = csv.reader(f)
    for row in reader:
        x.append(float(row[0]))
        y.append(float(row[1]))

# print(type(x[0]))
# print(x)

# Checking the loss function implementation
theta1=[0,0]
theta2=[-1,20]

if abs(loss_func(x,y,theta1)-1806.551)<0.001:
    print("***PASS***: Loss function check for theta1=(0,0) passed!")
else:
    print("***FAIL***: Loss function check for theta1=(0,0) failed!")

if abs(loss_func(x,y,theta2)-330.099)<0.001:
    print("***PASS***: Loss function check for theta2=(-1,20) passed!")
else:
    print("***FAIL***: Loss function check for theta1=(-1,20) failed!")


# Implementing Gradient Descent
m=len(x)
theta=[100,-10]
J=[]
for iter in range(n):
    for i in range(m):
        theta[0]+=alpha*loss_func_der(x,y,theta)*1
        # print(loss_func_der(x,y,theta))
        theta[1]+=alpha*loss_func_der(x,y,theta)*x[i]

    print(theta)
    J.append(loss_func(x,y,theta))

print(loss_func(x,y,theta))
x1=np.array(x)
y1=np.array(y)

x1t=np.transpose(x1)
print(x1t.dot(y1))



######## PLOTTING ########
# Plotting the raw data
plt.subplot(121)
plt.plot(x,y,'bx')
plt.xlabel('House living areas in 1000 square feet')
plt.ylabel('House prices in 10,000 dollars')


# Plotting loss function vs 
plt.subplot(122)
plt.plot(range(1,n+1),J)

plt.show()