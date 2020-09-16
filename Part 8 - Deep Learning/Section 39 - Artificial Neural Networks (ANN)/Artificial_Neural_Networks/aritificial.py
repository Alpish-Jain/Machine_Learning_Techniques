#part-1 --Data preprocessing
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

#Feature Scaling(It's very compulsory in neural networks so one variable dominates any other)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#part-2 building the ANN

#importing the keras and packages
import keras
#using sequential modeule to initialize our neural network and dense module to build the layers of ANN
from keras.models import Sequential
from keras.layers import Dense

#initializing the ANN
neural_classifier=Sequential()
#adding the input and the first hidden layer
neural_classifier.add(Dense(output_dim=6,init="uniform",activation="relu",input_dim=11))#no of nodes in the input layer is the number of dependent variable anad the ouput layer has one node since the result is binary
##the number of nodes in hidden layer are the average of nodes in input and output layer
#adding more hidden layer
neural_classifier.add(Dense(output_dim=6,init="uniform",activation="relu"))#no need of input dim now since now we have specified the input layer nodes in the previous layer
#adding the output layer
neural_classifier.add(Dense(output_dim=1,init="uniform",activation="sigmoid"))
#Compiling the ANN(applying stochastic gradient descend)
neural_classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])#loss function is like OLS etc.

#fitting ANN to the training set
neural_classifier.fit(X_train,y_train,batch_size=10,nb_epoch=100)

#predicting the results
y_pred=neural_classifier.predict(X_test)
y_pred=(y_pred>0.5)

#checking the accuracy using confusion metrics
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)