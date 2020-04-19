
######## TITLE ########
# Developed by: Saber Nemati
# email: mnemat2@lsu.edu


# Importing packages
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt


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

dir(plt)
