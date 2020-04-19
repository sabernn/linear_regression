
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
        y_hat=theta[0]+theta[1]*inp[i]
        y=out[i]
        J+=(y_hat-y)**2

    return J/2/m

# Declaring the update term function
def loss_func_der(inp,out,theta,alpha):
    m=len(inp)
    dJ=[0,0]
    
    for i in range(m):
        y_hat=theta[0]+theta[1]*inp[i]
        y=out[i]
        dJ[0]+=alpha*(y-y_hat)*1/m
        dJ[1]+=alpha*(y-y_hat)*x[i]/m

    return dJ


########     INPUT DATA    ########
# Number of iterations
n=50
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
theta=[0,0]
J=[]
for iter in range(n):
    theta[0]+=loss_func_der(x,y,theta,alpha)[0]
    theta[1]+=loss_func_der(x,y,theta,alpha)[1]

    J.append(loss_func(x,y,theta))

print(loss_func(x,y,theta))



######## PLOTTING ########
# Plotting the raw data
plt.subplot(121)
plt.plot(x,y,'bx')
plt.xlabel('House living areas in 1000 square feet')
plt.ylabel('House prices in 10,000 dollars')
plt.grid(axis='both')


# Plotting loss function vs 
plt.subplot(122)
plt.plot(range(1,n+1),J)
plt.xlabel('Iterations')
plt.ylabel('Loss value (J)')
plt.grid(axis='both')
plt.show()