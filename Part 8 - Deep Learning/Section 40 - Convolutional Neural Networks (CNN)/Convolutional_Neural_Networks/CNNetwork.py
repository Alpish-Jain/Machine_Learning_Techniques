# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 19:06:53 2019

@author: Alpish
"""
'''preprocessing was done manually and dataset was splitted to training and test set with associated labels as 
cats or dogs'''
#Part-1-Building the CNN

#importing the keras packages and libraries
from keras.models import Sequential#inititalize the NN
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#Initializing the neural network
classifier=Sequential()
#adding the different layers
#step-1 Convolution
classifier.add(Convolution2D(32,(3,3),activation='relu',input_shape=(64,64,3)))#no of channels=3,dim of 2d array=64
#step-2 Pooling(we keep the spatial features and the size of image is reduced)
classifier.add(MaxPooling2D(pool_size=(2,2)))
#adding 2nd convolutional layer although we can add full connection also
classifier.add(Convolution2D(32,(3,3),activation='relu'))#no of channels=3,dim of 2d array=64
#applying max pooling again
classifier.add(MaxPooling2D(pool_size=(2,2)))

#step-3 flattening(taking pooled feature maps and put them in a single vector(this will be the input layer))
classifier.add(Flatten())#since we have a classifier keras understands that it needs to flatten the previous layer so yayy..no arguments!!
#step-4 Full connection(making classic ANN as previous)
classifier.add(Dense(activation='relu',units=128))#for hidden layer
classifier.add(Dense(activation='sigmoid',units=1))#for output layer
#Compiling the CNN(choosing stochastic gradient descend,loss function etc)
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#part-2 fitting the CNN to the images
#trick:-Image augmentation(the training set of images is divided into different batches where eah batch is different from another i.e. one batch images are rotated,flipping,shifting them etc. therefore a lot more material to train )
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,#to keep pixel values in the range of 0 and 1
        shear_range=0.2,
        zoom_range=0.2,#apply random zooms
        horizontal_flip=True)#images will be flipped

test_datagen = ImageDataGenerator(rescale=1./255)
#applying image data gen to images in the training set
training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),#size of images expected in CNN model
        batch_size=32,
        class_mode='binary')#no of class label attributes

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000/32,#no of images in training set(The input here will be samples per epoch/batch size i.e 8000/32)
        epochs=25,
        validation_data=test_set,
        validation_steps=2000/32)#images in test set
