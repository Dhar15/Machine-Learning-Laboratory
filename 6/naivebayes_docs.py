''' Q.6) Assuming a set of documents that need to be classified, use the naïve Bayesian Document Classifier model to perform this task. 
         Built-in Java classes/API can be used to write the program.
         Calculate the accuracy, precision, and recall for your data set. '''

import pandas as pd

msg = pd.read_csv('naivetext1.csv', names=['message','label'])
print("The dimensions of dataset are: ", msg.shape)

# map positive to 1, negative target value to 0
msg['labelnum'] = msg.label.map({'pos':1, 'neg':0})
X = msg.message
y = msg.labelnum

# Split the dataset into train and test data
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(X, y)
# =============================================================================
# print(xtrain.shape)
# print(xtest.shape)
# print(ytrain.shape)
# print(ytest.shape)
# print("\n Training Data- \n")
# print(xtrain)
# =============================================================================

# Convert a collection of text documents to a matrix of token counts
# Output of count vectorizer is a sparse matrix
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
xtrain_dtm = count_vect.fit_transform(xtrain)
xtest_dtm = count_vect.transform(xtest)
# get_feature_names() returns a list of feature names
print("\n",count_vect.get_feature_names())

df = pd.DataFrame(xtrain_dtm.toarray(), columns=count_vect.get_feature_names())
print("\n",df)               # Tabular Representation
# print("\n",xtrain_dtm)     # Sparse Matrix Representation

# Training Naive Bayes(NB) classifier on the training data
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(xtrain_dtm, ytrain)
predicted = clf.predict(xtest_dtm)

# Print accuracy metrics
from sklearn import metrics
print("\n -------- Accuracy Metrics --------- ")
accuracy = metrics.accuracy_score(ytest,predicted)
cf = metrics.confusion_matrix(ytest, predicted)
recall = metrics.recall_score(ytest, predicted)
precision = metrics.precision_score(ytest, predicted)
print("Accuracy of classifier is: ", accuracy)
print("Confusion matrix: \n", cf)
print("Recall: ", recall)
print("Precision: ", precision)
print(" ----------------------------------- ")
