# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 17:28:21 2019

@author: Alpish
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

'''# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)'''

#fitting linear regression to the dataset
from sklearn.linear_model import LinearRegression
Linreg=LinearRegression()
Linreg.fit(X,y)

#fitting polynomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
x_poly=poly_reg.fit_transform(X)#fit model to x then transform x
poly_reg.fit(x_poly,y)
linreg2=LinearRegression()
linreg2.fit(x_poly,y)

#visualizing the results of linear regression
plt.scatter(X,y,color='k')
plt.plot(X,Linreg.predict(X),color='red')
plt.xlabel("level")
plt.ylabel('Salaries')
plt.title('Linear regression')
plt.show()
#visualizing the results of Polynomial linear regression
#to increment the step size of the x values
X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape((len(X_grid),1))
#just main visualizing steps
plt.scatter(X,y,color='k')
plt.plot(X,linreg2.predict(poly_reg.fit_transform(X)),color='red')
plt.xlabel("level")
plt.ylabel('Salaries')
plt.title('Polynomial regression')
plt.show()

#predicting a result with linearr regression
Linreg.predict(6.5)
#predicting a result with polynomial regression
linreg2.predict(poly_reg.fit_transform(6.5))
