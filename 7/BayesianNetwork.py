''' Q.7) Write a program to construct a Bayesian network considering medical data. Use this model to demonstrate the diagnosis of heart patients using 
         standard Heart Disease Data Set. You can use Java/Python ML library classes/API.'''
         
# A Bayesian Network is a type of Graphical Model that represents a set of variables and their conditional
# dependencies via a directed acyclic graph (DAG). 
         
# pgmpy is a python library for working with Probabilistic Graphical Models. It allows the user to create their own 
# graphical models and answer inference or map queries over them. pgmpy has implementation of many inference 
# algorithms like VariableElimination, Belief Propagation etc.
         
import pandas as pd
import numpy as np
import csv
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination

# Read the attributes
lines = list(csv.reader(open('heart.csv', 'r')))
attributes = lines[0]     # Headers
print(attributes)

# Read the Cleveland Heart Disease data
heartDisease = pd.read_csv('heart.csv', names= attributes)
heartDisease = heartDisease.replace('?', np.nan)              # Handling missing values

# View the data
print('Few examples from the dataset are given below- ')
print(heartDisease.head())
print('\nAttributes and data types-')
print(heartDisease.dtypes)

# Model a Bayesian Network
model = BayesianModel([('age', 'heartdisease'), ('fbs', 'heartdisease'), ('sex', 'heartdisease'),
('cp', 'heartdisease'), ('trestbps', 'heartdisease'),
('heartdisease', 'restecg'), ('heartdisease', 'thalach'),
('heartdisease', 'chol')])

# Learning CPD's (Conditional Probability Distribution) using Maximum Likelihood Estimators
print('\nLearning CPDs using Maximum Likelihood Estimators...')
model.fit(heartDisease, estimator=MaximumLikelihoodEstimator)

#Deducing with Bayesian Network
print('\nInferencing with Bayesian Network:')
HeartDisease_infer = VariableElimination(model)

print('\n1.Probability of HeartDisease given Age = 20') 
q = HeartDisease_infer.query(variables=['heartdisease'], evidence={'age': 20})
print(q)

print('\n2. Probability of HeartDisease given cp (Chest pain) = 2')
q = HeartDisease_infer.query(variables=['heartdisease'], evidence={'cp': 2})
print(q)
