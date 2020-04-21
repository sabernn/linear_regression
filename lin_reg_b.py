
######## TITLE ########
# CSC 7333 Programming Assignment - Part (b)
# Instructor: Dr. Jianhua Chen
# Developed by: Saber Nemati
# email: mnemat2@lsu.edu


########     INPUT DATA    ########
# Number of iterations
n=50
# Learning rate
alpha=1.5
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
        y_hat=theta[0]*inp[0][i]+theta[1]*inp[1][i]+theta[2]*inp[2][i]+theta[3]*inp[3][i]
        y=out[i]
        J+=(y_hat-y)**2

    return J/2/m

# Declaring the update term function
def loss_func_der(inp,out,theta,alpha):
    m=len(inp)
    dJ=[0,0,0,0]
    
    for i in range(m):
        y_hat=theta[0]+theta[1]*inp[1][i]+theta[2]*inp[2][i]+theta[3]*inp[3][i]
        y=out[i]
        dJ[0]+=alpha*(y-y_hat)*inp[0][i]/m
        dJ[1]+=alpha*(y-y_hat)*inp[1][i]/m
        dJ[2]+=alpha*(y-y_hat)*inp[2][i]/m
        dJ[3]+=alpha*(y-y_hat)*inp[3][i]/m

    return dJ


# Declaring standardization function
def standardize(x):
    m=np.average(x)
    std=np.std(x)

    return (x-m)/std

# Declaring single input standardization function
def standardize_s(x0,x):
    m=np.average(x)
    std=np.std(x)

    return (x0-m)/std

# Declaring destandardization function
def destandardize(xs):
    m=np.average(xs)
    std=np.std(xs)

    return (std*xs+m)

# Declaring single input destandardization function
def destandardize_s(x0,xs):
    m=np.average(xs)
    std=np.std(xs)

    return (std*x0+m)

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
print("--------------------------------------------------------------------------------------------")
print("(1) First 5 rows of raw input data are: ")
for i in range(5):
    print("{0}\t{1}\t{2}\t{3}".format(x1[i],x2[i],x3[i],y[i]))


# Print out the first 5 rows of standardized data

x1s=standardize(x1)
x2s=standardize(x2)
x3s=standardize(x3)

xs=[[1.0]*len(x1s),x1s,x2s,x3s]
print("--------------------------------------------------------------------------------------------")
print("(2) First 5 rows of standardized data are: ")
for i in range(5):
    print("1\t{0}\t{1}\t{2}\t{3}".format(x1s[i],x2s[i],x3s[i],y[i]))

print("--------------------------------------------------------------------------------------------")
# Checking the loss function implementation
theta1=[0,0,0,0]
print("The cost (J) value for theta = {0} is: J = {1}".format(theta1,loss_func(xs,y,theta1)))


# Implementing Gradient Descent
m=len(xs)
theta=[0,0,0,0]
J=[]
print("--------------------------------------------------------------------------------------------")
print("Theta \t\t\t\t\t\t\t\t Loss Function (J)")
print("--------------------------------------------------------------------------------------------")
for iter in range(n):
    theta[0]+=loss_func_der(xs,y,theta,alpha)[0]
    theta[1]+=loss_func_der(xs,y,theta,alpha)[1]
    theta[2]+=loss_func_der(xs,y,theta,alpha)[2]
    theta[3]+=loss_func_der(xs,y,theta,alpha)[3]

    # Printing the values of theta and J for each iteration
    Jtemp=loss_func(xs,y,theta)
    J.append(Jtemp)
    theta_r=[round(theta[0],5),round(theta[1],5),round(theta[2],5),round(theta[3],5)]
    Jtemp=round(Jtemp,5)
    print("{0} \t\t {1}".format(theta_r,Jtemp))
print("--------------------------------------------------------------------------------------------")

# Prediction
xtests=[0,0,0,0]
xtests[0]=1
n_bed=3
xtests[1]=standardize_s(n_bed,x1)
liv_area=2000
xtests[2]=standardize_s(liv_area,x2)
lot_area=8550
xtest=[n_bed,liv_area,lot_area]
xtests[3]=standardize_s(lot_area,x3)
price=round((theta[0]+theta[1]*xtests[1]+theta[2]*xtests[2]+theta[3]*xtests[3]),2)

print("Aproximate house price for {0} is: ${1}".format(xtest,price))
print("--------------------------------------------------------------------------------------------")


######## PLOTTING ########
# Plotting loss function vs 
plt.title("J vs n")
plt.plot(range(1,n+1),J)
plt.xlabel('Iterations')
plt.ylabel('Loss value (J)')
plt.grid(axis='both')
plt.show()