''' Q.4) Build an Artificial Neural Network by implementing the Backpropagation algorithm and test the same using appropriat
         data sets. '''
         
import numpy as np
X = np.array(([2,9],[1,5],[3,6]), dtype=float)    # Input values
y = np.array(([92],[86],[89]), dtype=float)       # Output percentages
X = X/np.amax(X,axis=0)                           # Maximum of X array longitudinally
y = y/100

#print(X)
#print(y)

#Variable Initialization
epoch = 7000                                      # Setting training iterations
lr = 0.1                                          # Setting learning rate
inputlayer_neurons = 2                            # No.of input features in dataset
hiddenlayer_neurons = 3                           # No.of hidden layer neurons
output_neurons = 1                                # No.of neurons at output layer

# Weight and Bias initialization
# np.random.uniform(x,y) draws a random range of numbers uniformly of dimension x*y
weight_hidden = np.random.uniform(size=(inputlayer_neurons, hiddenlayer_neurons))   # Weights towards hidden layer
bias_hidden = np.random.uniform(size=(1, hiddenlayer_neurons))                      # Bias weights of hidden layer
weight_output = np.random.uniform(size=(hiddenlayer_neurons, output_neurons))       # Weights towards output layer
bias_output = np.random.uniform(size=(1, output_neurons))                           # Bias weights of ouput layer

#Sigmoid Function - maps any value between 0 and 1
def sigmoid(x):
  return 1/(1 + np.exp(-x))

#Derivative of sigmoid function
def derivates_sigmoid(x):
  return x * (1-x)

# FORWARD PROPOGATION - Propogate the input forward through the network
# Input the instance to the network and compute output of every unit in the network
for i in range(epoch):
  hinp1 = np.dot(X, weight_hidden)    # Σ XiWi + Bi
  hinp = hinp1 + bias_hidden             
  hlayer_act = sigmoid(hinp)          # Apply activation function to generate hidden layer values (which are i/p to output layer)
  
  outinp1 = np.dot(hlayer_act, weight_output)
  outinp = outinp1 + bias_output
  output = sigmoid(outinp)            # The predicted output values
  
  # To stabilize our neural network, we have to backpropogate - calculate error terms and update weight values
  
  # BACKPROPOGATION 
  ''' Propogate the errors backward through the network
      For each network output unit k, calculate its error term 𝛿k
      𝛿k <- ok(l - ok)(tk - ok) '''
  EO = y - output                           # Target class (Actual) values - Predicted output values
  out_grad = derivates_sigmoid(output)        
  del_k = EO * out_grad                     # Error term at output
  
  ''' For each hidden layer h, calculate the error term 𝛿h
      𝛿h <- oh(l - oh) ΣWkh*𝛿k '''
  EH = del_k.dot(weight_output.T)
  hidden_grad = derivates_sigmoid(hlayer_act)
  del_h = EH * hidden_grad                  # Error term at hidden layer
  
  # Finally, Upgrade each network weight
  # Wji <- Wji + η * 𝛿j * Xji , where η is the learning rate
  weight_output += hlayer_act.T.dot(del_k) * lr      # Dot product of next layer error and current layer output
  weight_hidden += X.T.dot(del_h) * lr   
  
# Print the resultant values
print("Input: \n" + str(X))
print("Actual Output: \n" + str(y))
print("Predicted Output: \n" + str(output))
  
  
