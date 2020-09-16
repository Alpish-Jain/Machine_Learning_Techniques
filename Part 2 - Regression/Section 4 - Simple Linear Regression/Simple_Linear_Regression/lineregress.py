# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 12:02:34 2019

@author: Alpish
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset=pd.read_csv(r"C:\Users\Alpish\Downloads\headbrain.csv")
X = dataset.iloc[:, 2:3].values
y = dataset.iloc[:, 3:].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

#fitting linear regression in training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

#predicting the results based on previous learning
y_pred=regressor.predict(X_test)

#visualizing the prediction
plt.scatter(X_test,y_test)
plt.plot(X_test,regressor.predict(X_test),color='k')
plt.xlabel("years of exp")
plt.ylabel("salary")
plt.title("linregress")
plt.show()
