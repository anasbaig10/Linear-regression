#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 23:03:32 2019

@author: anas
"""

# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('/Users/anas/Downloads/Machine Learning A-Z New-2/Part 2 - Regression/Section 4 - Simple Linear Regression/Salary_Data.csv')
#independent variable is X
X = dataset.iloc[:, :-1].values
#Dependent Variable is Y
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Feature Scaling ie reducing the values to compute
"""from sklearn.preprocessing import StandardScaler
sc_X =StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.fit_transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))"""


#Fitting simple linear regression to training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
#fitting regressor to training set and independent then dependent
regressor.fit(X_train,y_train)
#predection of test set
y_pred=regressor.predict(X_test) 

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()