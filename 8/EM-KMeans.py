''' Q.8) Apply EM algorithm to cluster a set of data stored in a .CSV file. Use the same data set for clustering using k-Means algorithm. Compare the results of these two 
         algorithms and comment on the quality of clustering. You can add Java/Python ML library classes/API in the program. '''
         
''' THEORY :
  Expectation-Maximization(EM) algorithm is used for latent variables(variables that are not directly observable and are actually inferred 
  from the values of the other observed variables) in order to predict their values with the condition that the general form of probability 
  distribution governing those latent variables is known to us. It is a statistical algorithm for finding the right model parameters.
  
  Gaussian Mixture Models (GMMs) are probablistic models which assume that there are a certain number of Gaussian distributions, and 
  each of these distributions represent a cluster. Hence, a Gaussian Mixture Model tends to group the data points belonging to a single distribution together.
  For a dataset with d features, we would have a mixture of k Gaussian distributions(where k is equivalent to the number of clusters), 
  each having a certain mean vector and variance matrix. For each Gaussian, the mean and variance values are calculated using the EM algorithm.
  
  K-means is a famous unsupervised centroid-based algorithm, or a distance-based algorithm, where we calculate the distances to assign 
  a point to a cluster. In K-Means, each cluster is associated with a centroid. '''
  
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

# Import the dataset
data = pd.read_csv("ex.csv")
print("Input data and shape-")
print(data.shape)
print(data.head())

# Getting the values and plotting it
f1 = data['V1'].values
print(f1)

f2 = data['V2'].values
print(f2)

X = np.array(list(zip(f1,f2)))   # An array of lists of combined f1 and f2 values   
print(X)

print('Graph for the whole dataset generated.')
plt.scatter(f1,f2,c='black',s=300)
plt.show()

# K MEANS CLUSTERING
kmeans = KMeans(2, random_state=0)
labels = kmeans.fit(X).predict(X)
print("Labels: ", labels)
centroids = kmeans.cluster_centers_
print("Centroids:", centroids)

print('Graph using KMeans clustering generated.')
plt.scatter(X[:,0],X[:,1], c=labels, s=40)
plt.scatter(centroids[:,0], centroids[:,1], marker='*', s=200, c='#050505')
plt.show()

# GMM
gmm = GaussianMixture(n_components=2).fit(X)
labels = gmm.predict(X)
print("GMM Labels-")
print(labels)
probs = gmm.predict_proba(X)        # Predict the posterior probablity of each component 
size = 10 * probs.max(1) ** 3

print("Graph using EM algorithm generated.")
# print(probs[:300].round(4))
plt.scatter(X[:,0], X[:,1], c=labels, s=size, cmap='viridis')
plt.show()
