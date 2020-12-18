''' Q.10) Implement the non-parametric Locally Weighted Regression algorithm in order to fit data points. Select appropriate data set for your experiment and draw graphs. '''
          
''' Locally weighted linear regression is a non-parametric supervised learning algorithm, that is, the model does not learn a fixed set of parameters 
    as is done in ordinary linear regression. Rather parameters θ are computed individually for each query point x. While computing 
    θ, a higher “preference” is given to the points in the training set lying in the vicinity of x than the points lying far away from x.'''
    
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def kernel(point, xmat, k):
  m,n = np.shape(xmat)
  weights = np.mat(np.eye((m)))     # eye() returns a 2-D array with ones on the diagonal and zeros elsewhere.
  for j in range(m):
    diff = point - X[j]
    weights[j,j] = np.exp(diff * diff.T / (-2.0 * k ** 2))
    
  return weights

def localWeight(point, xmat, ymat, k):
  weight = kernel(point, xmat, k)
  W = (X.T * (weight*X)).I * (X.T * (weight * ymat.T))
  return W

def localWeightRegression(xmat, ymat, k):
  m,n = np.shape(xmat)
  ypred = np.zeros(m)
  for i in range(m):
    ypred[i] = xmat[i] * localWeight(xmat[i], xmat, ymat, k)
    
  return ypred

# Load the dataset
data = pd.read_csv("tips.csv")
bill = np.array(data.total_bill)
tip = np.array(data.tip)

mbill = np.mat(bill)               # Interpret the input as a matrix.
mtip = np.mat(tip)
m = np.shape(mbill)[1]

ones = np.mat(np.ones(m))          # Matrix of 1's of same shape.

X = np.hstack((ones.T, mbill.T))   # Stack arrays in sequence horizontally (column wise).

yPred = localWeightRegression(X, mtip, 10)

sortIndex = X[:,1].argsort(0)
xSort = X[sortIndex][:,0]

# Plot the regression graph to view results
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(bill, tip, color = 'green')
ax.plot(xSort[:,1], yPred[sortIndex], color='red', linewidth=5)
plt.xlabel('Total bill')
plt.ylabel('Tip')
plt.show()
