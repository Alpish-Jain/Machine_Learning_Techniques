# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 16:24:57 2019

@author: Alpish
"""
#Natural Language Processing
#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
dataset=pd.read_csv("Restaurant_Reviews.tsv",delimiter='\t',quoting=3)

#1.cleaning the texts
corpus=[]
for i in range(0,1000):
    #step-1 is to enter what letters we dont want to remove(stemming)
    import re
    new_review=re.sub('[^a-zA-Z]',' ', dataset['Review'][i])
    #step-2 is to convert the review to lowercase only
    new_review=new_review.lower()
    #step-3 removing the unnecessary words like this,that,it etc.otherwise they will have their own column 
    import nltk
    from nltk.corpus import stopwords
    nltk.download('stopwords')
    new_review=new_review.split()
    #stemming-keeping only the root words of the words like love for loved,loving,loves etc.
    from nltk.stem.porter import PorterStemmer
    ps=PorterStemmer()
    new_review=[ps.stem(word) for word in new_review if not word in set(stopwords.words('english'))]
    #joining the words
    new_review=' '.join(new_review)
    corpus.append(new_review)

#creating the bag of words model(we use max features here but we can also use dimensionality reduction)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)##the parameters contain the parameters required for the above tasks
X=cv.fit_transform(corpus).toarray()
y=dataset.iloc[:,1:2].values

#training our classification model based on sparse matrix X
#splitting into train and test data
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)

#fitting naive bayes to the training set
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(X_train,y_train)

#predicting the results
y_pred=classifier.predict(X_test)

#making the confusion matrix to check classification accuracy
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)









