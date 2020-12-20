# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 15:05:50 2020

@author: kshit
"""

''' Q.3) Write a program to demonstrate the working of the decision tree based ID3 algorithm. Use
         an appropriate data set for building the decision tree and apply this knowledge to classify a
         new sample. '''
         
import numpy as np
import math
import csv

def read_data(filename):
  with open(filename, 'r') as csvfile:
    datareader =  csv.reader(csvfile, delimiter=',')
    headers = next(datareader)
    metadata = []           # For the headers
    traindata = []          # For the data rows
    for name in headers:
      metadata.append(name)
    for row in datareader:
      traindata.append(row)
      
    return (metadata,traindata)

class Node:
  def __init__(self, attribute):
    self.attribute = attribute
    self.children = []
    self.answer = ""
    
  def __str__(self):
    return self.attribute
  
# Function to return array of unique target variables along with all the rows belonging to a specific variable in dict
def subtables(data, col, delete):
  dict = {}
  items = np.unique(data[:, col])
  count = np.zeros((items.shape[0], 1), dtype=np.int32)
  
  for x in range(items.shape[0]):
    for y in range(data.shape[0]):
      if data[y, col] == items[x]:
        count[x] += 1                       # Count of each class of each attribute
        
  for x in range(items.shape[0]):
    ''' dict holds all the values of a particular target class. Ex-
    dict[items[2]] = dict['rain'] = array([  [b'rain', b'mild', b'high', b'weak', b'yes'],
                                             [b'rain', b'cool', b'normal', b'weak', b'yes'],
                                             [b'rain', b'cool', b'normal', b'strong ', b'no'],
                                             [b'rain', b'mild ', b'normal', b'weak', b'yes'],
                                             [b'rain', b'mild', b'high', b'strong ', b'no']], dtype='|S32')] '''
                                           
    dict[items[x]] = np.empty((int(count[x]), data.shape[1]), dtype="|S32")     # np.empty() returns a new unitialized array of given shape and type. S32 is string datatype.

    pos = 0
    for y in range(data.shape[0]):
      if data[y, col] == items[x]:
        dict[items[x]][pos] = data[y]
        pos += 1
        
    if delete:
      dict[items[x]] = np.delete(dict[items[x]], col, 1)
      
  return items, dict

# Calculate the entropy of attributes
def entropy(S):
  items = np.unique(S)
  if items.size == 1:
    return 0
  
  counts = np.zeros((items.shape[0], 1))
  sums = 0
  
  for x in range(items.shape[0]):
    counts[x] = sum(S == items[x]) / (S.size * 1.0)
    
  for count in counts:
    sums += -1 * count * math.log(count,2) 
    
  return sums

# Calculate the info gain of each attribute
def gain_ratio(data, col):
  items, dict = subtables(data, col, delete=False)
  total_size = data.shape[0]
  entropies = np.zeros((items.shape[0], 1))
  intrinsic = np.zeros((items.shape[0], 1))
  
  for x in range(items.shape[0]):
    ratio = dict[items[x]].shape[0] / (total_size * 1.0)
    entropies[x] = ratio * entropy(dict[items[x]][:,-1])
    intrinsic[x] = ratio * math.log(ratio, 2)
    
  total_entropy = entropy(data[:,-1])
  iv = -1 * sum(intrinsic)
  
  for x in range(entropies.shape[0]):
    total_entropy -= entropies[x]
    
  return total_entropy / iv

def create_node(data, metadata):
  if (np.unique(data[:,-1])).shape[0] == 1:         # If there is only one target class. np.unique() finds the unique elements of an array.
    node = Node("")
    node.answer = np.unique(data[:,-1])[0]
    return node
  
  gains = np.zeros((data.shape[1] - 1, 1))
  for col in range(data.shape[1] - 1):
    gains[col] = gain_ratio(data, col)
  
  split = np.argmax(gains)
  node = Node(metadata[split])
  metadata = np.delete(metadata, split, 0)
  items, dict = subtables(data, split, delete=True)
  
  for x in range(items.shape[0]):
    child = create_node(dict[items[x]], metadata)
    node.children.append((items[x], child))
  return node

# For spacing of the tree
def empty(size):
  s = ""
  for x in range(size):
    s += "   "
  return s

# Recursive function to print the decision tree
def print_tree(node,level):
  if node.answer != "":
    print(empty(level), node.answer)
    return
  
  print(empty(level), node.attribute)
  
  for value, n in node.children:
    print(empty(level+1), value)
    print_tree(n, level+2)
  
  
metadata, traindata = read_data("tennis.csv")
# Storing rows as array elements
data = np.array(traindata)       
node = create_node(data, metadata)
print_tree(node, 0)
