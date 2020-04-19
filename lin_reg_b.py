
######## TITLE ########
# CSC 7333 Programming Assignment - Part (b)
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
import numpy as np
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


# Declaring standardization function
def standardize(x):
    m=np.average(x)
    std=np.std(x)

    return (x-m)/std


# Declaring normalization function
def normalize(x):
    xmin=np.min(x)
    xmax=np.max(x)

    return (x-xmin)/(xmax-xmin)

# Initializing input and output vectors
x1=[]
x2=[]
x3=[]
y=[]
# Importing data
with open('data//KCSmall_NS2.csv', newline='') as f:
    reader = csv.reader(f)
    for row in reader:
        x1.append(float(row[0]))
        x2.append(float(row[1]))
        x3.append(float(row[2]))
        y.append(float(row[3]))


# Print first 5 rows of the raw input data
print("First 5 rows of raw input data are: ")
for i in range(5):
    print("{0}\t{1}\t{2}\t{3}".format(x1[i],x2[i],x3[i],y[i]))

# Print out the first 5 rows of normalized data
x1n=normalize(x1)
x2n=normalize(x2)
x3n=normalize(x3)
yn=normalize(y)
print("First 5 rows of normalized data are: ")
for i in range(5):
    print("{0}\t{1}\t{2}\t{3}".format(x1n[i],x2n[i],x3n[i],yn[i]))


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

xtest=[3.5,7]
for t in xtest:
    liv_area=t*1000
    price=(theta[0]+theta[1]*t)*10000
    print("Aproximate house prices for {0} square feet living area is: ${1}".format(liv_area,price))


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