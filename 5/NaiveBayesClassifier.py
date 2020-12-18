# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 16:18:20 2020

@author: kshit
"""

''' Q.5) Write a program to implement the naïve Bayesian classifier for a sample training data set
         stored as a .CSV file. Compute the accuracy of the classifier, considering few test data sets. '''
         
import numpy as np
import pandas as pd

mush =  pd.read_csv("mushrooms.csv")        # Shape = (65*20)
mush.replace('?', np.nan, inplace=True)     # Handling missing data
print("Initially,", len(mush.columns), "columns. After dropping NA values -", len(mush.dropna(axis=1).columns), "columns")

#Drop wherever you have ? as the values are not known
mush.dropna(axis=1,inplace=True)

#The first column in dataset is 'class' which is the target variable
target = 'class'
features = mush.columns[mush.columns != target]       # Independent attributes
# print(features)
classes = mush[target].unique()                       # Target variable classes - 'p' and 'e'
# print(classes)
test = mush.sample(frac=0.3)                          # Save 30%(of 65 = 20) random sample data as the test set
# print(test)
mush = mush.drop(test.index)                          # Drop the random test rows generated previously(Now shape is 45*20), this is the training set
# print(mush)

probs = {}
probcl = {}

''' 
#df = pd.DataFrame({'mycolumn': [1,2,2,2,3,3,4]})
#for val, cnt in df.mycolumn.value_counts().iteritems():
#print 'value', val, 'was found', cnt, 'times'
#value 2 was found 3 times
#value 3 was found 2 times
#value 4 was found 1 times
#value 1 was found 1 times
'''
for x in classes:
  mushcl = mush[mush[target] == x][features]
  # print(mushcl)      # All rows and columns of a particular target class
  clsp = {}
  tot = len(mushcl)
  # print(tot)        # No.of rows that target class consists of
  for col in mushcl.columns:
    colp = {}
    for val, cnt in mushcl[col].value_counts().iteritems():
      pr = cnt / tot
      # print(pr)
      colp[val] = pr
      clsp[col] = colp
      
  probs[x] = clsp
  probcl[x] = len(mushcl)/len(mush)                   # Probablity of each class
  
def probabs(x):
  if not isinstance(x, pd.Series):
    raise IOError("Arg must be of type Series")
    
  probab = {}
  
  for cl in classes:
    pr = probcl[cl]
    for col, val in x.iteritems():
      try:
        pr *= probs[cl][col][val]
      except KeyError:
        pr = 0
    probab[cl] = pr
  return probab

def classify(x):
  probab = probabs(x)
  mx = 0
  mxcl = ""
  
  for cl, pr in probab.items():
    if pr > mx:
      mx = pr
      mxcl = cl
      
  return mxcl

# Train data
b = []
for i in mush.index:
  # print(classify(mush.loc[i,features]),mush.loc[i,target])
  b.append(classify(mush.loc[i,features]) == mush.loc[i,target])
print(sum(b),"correct out of",len(mush))
print("Accuracy:", sum(b)/len(mush))

#Test data
b = []
for i in test.index:
  #print(classify(mush.loc[i,features]),mush.loc[i,target])
  b.append(classify(test.loc[i,features]) == test.loc[i,target])
print(sum(b),"correct out of",len(test))
print("Accuracy:",sum(b)/len(test))

      
