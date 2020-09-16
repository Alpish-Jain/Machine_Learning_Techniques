# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 20:18:36 2020

@author: Alpish
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
from sklearn.compose import ColumnTransformer
transformer=ColumnTransformer([('one_hot_encoder',OneHotEncoder(),[1])],remainder='passthrough')
X=np.array(transformer.fit_transform(X),dtype=np.float)
X=X[:,1:]

#Splitting the dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#feature scaling is already done in XGBoost(with high performance,great execution speed and we can keep the interpretation of the model)
#fitting XGBoost to the training set
from xgboost import XGBClassifier
classifier=XGBClassifier()
classifier.fit(X_train,y_train)

#predicting the test set results
y_pred=classifier.predict(X_test)

#using confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

#using k fold cross validation
from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=100)
accuracies.mean()
