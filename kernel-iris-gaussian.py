# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 10:56:50 2019

@author: Pablo
"""

import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd  


# load data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# Assign colum names to the dataset
colnames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

# Read dataset to pandas dataframe
irisdata = pd.read_csv(url, names=colnames) 


# divide the data into attributes and labels
X = irisdata.drop('Class', axis=1)  
y = irisdata['Class']  


# training and test split
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)  


# train the algorithm: Gaussian kernel
from sklearn.svm import SVC  
svclassifier = SVC(kernel='rbf')  
svclassifier.fit(X_train, y_train)  

# predict
y_pred = svclassifier.predict(X_test)  

# evaluate
from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))  