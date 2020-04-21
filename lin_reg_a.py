
######## TITLE ########
# CSC 7333 Programming Assignment - Part (a)
# Instructor: Dr. Jianhua Chen
# Developed by: Saber Nemati
# email: mnemat2@lsu.edu


########     INPUT DATA    ########
# Number of iterations
n=50
# Learning rate
alpha=0.1
######## END OF INPUT DATA ########


# Importing packages
# import numpy as np
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
        dJ[1]+=alpha*(y-y_hat)*inp[i]/m

    return dJ




# Initializing input and output vectors
x=[]
y=[]
# Importing data
with open('data//KCSmall2.csv', newline='') as f:
    reader = csv.reader(f)
    for row in reader:
        x.append(float(row[0]))
        y.append(float(row[1]))


# Checking the loss function implementation
theta1=[0,0]
theta2=[-1,20]

if abs(loss_func(x,y,theta1)-1806.551)<0.001:
    print("***PASS***: Loss function implementation check for theta1=(0,0) passed!")
else:
    print("***FAIL***: Loss function implementation check for theta1=(0,0) failed!")

if abs(loss_func(x,y,theta2)-330.099)<0.001:
    print("***PASS***: Loss function implementation check for theta2=(-1,20) passed!")
else:
    print("***FAIL***: Loss function implementation check for theta1=(-1,20) failed!")


# Implementing Gradient Descent
m=len(x)
theta=[100,100]
J=[]
print("----------------------------------------------------")
print("Theta \t\t\t Loss Function (J)")
print("----------------------------------------------------")
for iter in range(n):
    theta[0]+=loss_func_der(x,y,theta,alpha)[0]
    theta[1]+=loss_func_der(x,y,theta,alpha)[1]

    # Printing the values of theta and J for each iteration
    Jtemp=loss_func(x,y,theta)
    J.append(Jtemp)
    theta_r=[round(theta[0],5),round(theta[1],5)]
    Jtemp=round(Jtemp,5)
    print("{0} \t\t {1}".format(theta_r,Jtemp))
print("----------------------------------------------------")
print("The final theta learned: theta = [{0}, {1}]".format(theta[0],theta[1]))
print("The corresponding loss function: J = {0}".format(J[-1]))
print("----------------------------------------------------")
xtest=[3.5,7]
for t in xtest:
    liv_area=round(t*1000,2)
    price=round((theta[0]+theta[1]*t)*10000,2)
    print("Aproximate house price for {0} square feet living area is: ${1}".format(liv_area,price))
print("----------------------------------------------------")

######## PLOTTING ########
# Plotting the raw data
plt.subplot(121)
plt.title("Raw data")
plt.plot(x,y,'bx')
plt.xlabel('House living areas in 1000 square feet')
plt.ylabel('House prices in 10,000 dollars')
plt.grid(axis='both')


# Plotting loss function vs iterations
plt.subplot(122)
plt.title("J vs n")
plt.plot(range(1,n+1),J)
plt.xlabel('Iterations')
plt.ylabel('Loss value (J)')
plt.grid(axis='both')
plt.show()