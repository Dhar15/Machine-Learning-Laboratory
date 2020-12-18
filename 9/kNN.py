# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 22:21:45 2020

@author: kshit
"""

''' Q.9) Write a program to implement k-Nearest Neighbour algorithm to classify the iris data set. Print both correct and wrong predictions. 
         Java/Python ML library classes can be used for this problem. '''
         
''' k-Nearest Neighbor Algorithm is the simplest supervised learning ML algorithm. K-NN algorithm assumes the similarity between the new data 
    and available cases and assigns the new case into the category that is most similar to the available categories.
    K-NN is a "non-parametric" algorithm, which means it does not make any assumption on underlying data.
    It is also called a "lazy learner algorithm" because it does not learn from the training set immediately, instead it stores the dataset 
    and at the time of classification, it performs an action on the dataset.'''
    
import numpy as np
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Load the iris dataset
iris = load_iris()

# The target classes are the three species -  0- 'setosa', 1- 'versicolor', 2- 'virginica'
print("\nIRIS FEATURES/ TARGET NAMES: \n", iris.target_names)

for i in range(len(iris.target_names)):
  print("\n[{0}]:[{1}]".format(i, iris.target_names[i]))
  
# The dataset contains a set of 150 records under five attributes - sepal length, sepal width, petal length, petal width and species.
print("\nIRIS DATA: \n", iris["data"])
    
X_train, X_test, y_train, y_test = train_test_split(iris["data"], iris["target"])

# print("Target values:", iris["target"])
# print("\nX_TRAIN: ", X_train)
# print("\nX_TEST: ", X_test)
# print("\nY_TRAIN: ", y_train)
# print("\nY_TEST: ", y_test)

# KNN CLASSIFICATION
knn = KNeighborsClassifier(n_neighbors=1)

# Fit the model
knn.fit(X_train, y_train)

for i in range(len(X_test)):
  x = X_test[i]
  x_new = np.array([x])
  pred = knn.predict(x_new)
  print("\nActual : {0} {1}, Predicted : {2} {3}".format(y_test[i], iris["target_names"][y_test[i]], pred, iris["target_names"][pred]))
  
print("\nTEST SCORE(ACCURACY) = {:.2f}".format(knn.score(X_test, y_test)))
