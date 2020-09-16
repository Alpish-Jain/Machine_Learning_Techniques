# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 20:38:15 2019

@author: Alpish
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

#encoding categorical data to create dummy variables
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder=LabelEncoder()
X[:,3]=labelencoder.fit_transform(X[:,3])
onehotencoder=OneHotEncoder(categorical_features=[3])#index of column we want to one hot encode
X=onehotencoder.fit_transform(X).toarray()

#avoiding the dummy variable trap
X=X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#fitting multiple linear regression model into training set 
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

#predicting the test set results
y_pred=regressor.predict(X_test)

#building the optimal model using backward elimination
import statsmodels.formula.api as sm
X=np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)

X_opt=X[:,[0,1,2,3,4,5]]
regressor_ols=sm.OLS(endog=y,exog=X_opt).fit()
regressor_ols.summary()
#removing x2
X_opt=X[:,[0,1,3,4,5]]
regressor_ols=sm.OLS(endog=y,exog=X_opt).fit()
regressor_ols.summary()
#removing x1
X_opt=X[:,[0,3,4,5]]
regressor_ols=sm.OLS(endog=y,exog=X_opt).fit()
regressor_ols.summary()
#removing x4
X_opt=X[:,[0,3,5]]
regressor_ols=sm.OLS(endog=y,exog=X_opt).fit()
regressor_ols.summary()
#removing x5 if you thorougly follow backward elimination
X_opt=X[:,[0,3,5]]
regressor_ols=sm.OLS(endog=y,exog=X_opt).fit()
regressor_ols.summary()


