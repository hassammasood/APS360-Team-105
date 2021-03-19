# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 12:27:55 2021

@author: Roméo Cabrera & Hassam Masood
"""
'''Cell 0: Importing Libraries'''
import numpy as np
import os
import time
import sklearn as sk
from sklearn import svm, metrics
from skimage.feature import hog
#%%
'''Cell 1: Loading in Data and Preprocessing for SKLearn'''
#%%
'''Cell 2: Define Train/Predict Function'''
def train_and_predict(train_input_features, train_outputs, prediction_features):
    """
    :param train_input_features: (numpy.array) A two-dimensional NumPy array where each element
                        is an array that contains: sepal length, sepal width, petal length, and petal width   
    :param train_outputs: (numpy.array) A one-dimensional NumPy array where each element
                        is a number representing the species of iris which is described in
                        the same row of train_input_features. 0 represents Iris setosa,
                        1 represents Iris versicolor, and 2 represents Iris virginica.
    :param prediction_features: (numpy.array) A two-dimensional NumPy array where each element
                        is an array that contains: sepal length, sepal width, petal length, and petal width
    :returns: (list) The function should return an iterable (like list or numpy.ndarray) of the predicted 
                        iris species, one for each item in prediction_features
    """   
    
    classif=svm.SVC(kernel='poly',decision_function_shape='ovo',random_state=0)
    classif.fit(train_input_features,train_outputs)
    
    output_class=np.array(classif.predict(prediction_features))
    
    return output_class


####
#Function that loops through directory and loads images as numpy array,
# as well as fetch labels for each

####
#Function that extracts features from images converted to array using hog
# for use in SVM classifier

y_pred = train_and_predict(train_input_features, train_labels, test_features)
if y_pred is not None:
    print(metrics.accuracy_score(test_labels, pred_labels))
