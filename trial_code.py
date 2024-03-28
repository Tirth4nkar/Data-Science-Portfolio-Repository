import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy import special

#Data Prerequisites
path = "PATH TO YOUR DATA"
categories = ['A LIST OF TARGET CLASSES']

""" 
  for each category in the categories list, we try to map the training image data
  with the index of that category and append the train_set with a tuple.
  
  so than we can iterate over it like a traditional pytorch dataloader. this method 
  can also be replaced with text data. The ideology is similar though.
"""
img_size = 80
train_set = []
for category in categories:
    path_animal = os.path.join(path, category)
    for img in os.listdir(path_animal):
        try:
            img_array = cv2.imread(os.path.join(path_animal, img), cv2.IMREAD_GRAYSCALE)
            new_img_array = cv2.resize(img_array, (img_size, img_size))
            flattened_img_array = new_img_array.reshape(img_size*img_size)
            train_set.append([flattened_img_array, categories.index(category)])
        except:
            continue
random.shuffle(train_set)


X_train = []
Y_train = []
for sample in train_set:
    X_train.append(sample[0])
    Y_train.append(sample[1])

X_train = (np.array(X_train).T)/255
Y_train = np.array(Y_train).reshape((1, np.array(Y_train).shape[0]))

def create_mini_batches(X:np.array, Y:np.array,
                        mini_batch_size:int)->list:
  """ 
    Divide the training data into mini batches
  """
  mini_batches = list()
  m = X.shape[1]
  num_mini_batches:int = m // mini_batch_size
  
  permutation = list(np.random.permutation(m))
  shuffled_X = X[:, permutation]
  shuffled_Y = Y[:, permutation]
  
  for i in range(num_mini_batches):
      select_X = shuffled_X[:, mini_batch_size*i : mini_batch_size*(i+1)]
      select_Y = shuffled_Y[:, mini_batch_size*i : mini_batch_size*(i+1)]
      mini_batch = (select_X, select_Y)
      mini_batches.append(mini_batch)
  
  """ 
    If there are odd (2m+1) number of observations,
    we append the data by manually indexing it's position.
    Since we took a rounded division, in case of odd number
    of training data, the last image/text is the one to get 
    missed.
  """
  if m % mini_batch_size != 0:
      last_X = shuffled_X[:, mini_batch_size*num_mini_batches:m]
      last_Y = shuffled_Y[:, mini_batch_size*num_mini_batches:m]
      last_mini_batch = (last_X, last_Y)
      mini_batches.append(last_mini_batch)
      
  return mini_batches

def initialize_parameters(layers_dims): 
    L = len(layers_dims) # number of layers (including input layer), in this case L=4.
    parameters = {}
    for l in range(1,L): # range(1,4).
        parameters['W' + str(l)] = np.random.randn(layers_dims[l],layers_dims[l-1]) * np.sqrt(2/layers_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layers_dims[l],1))
    return parameters

def sigmoid(Z):
    A = special.expit(Z)
    return A,Z

def relu(Z):
    A = np.maximum(0.01*Z, Z)
    return A,Z

def forward_propagation(X, parameters):

    caches = [] #list containing Z for every node
    A = X
    L = int(len(parameters)/2)
    
    for l in range(1,L):
        A_prev = A
        W = parameters['W'+str(l)]
        b = parameters['b'+str(l)]
        Z = np.dot(W, A_prev) + b
        A, activation_cache = relu(Z) #activation_cache contains z[l].
        linear_cache = (A_prev, W, b) #linear_cache contains A[l-1], W[l], b[l].
        cache = (linear_cache, activation_cache)
        caches.append(cache)
    
    W = parameters['W'+str(L)]
    b = parameters['b'+str(L)]
    Z = np.dot(W, A) + b
    AL, activation_cache = sigmoid(Z)
    linear_cache = (A, W, b)
    cache = (linear_cache, activation_cache)
    caches.append(cache)
    
    return AL, caches

epsilon = 1e-8
def compute_cost(AL, Y, parameters, lambd):
    m = Y.shape[1] # number of examples
    L = int(len(parameters)/2) #[6400,100,20,1] L=3 (0,1,2)
    reg_cost = 0
    
    for l in range(L):
        W = parameters['W' + str(l+1)]
        reg_cost += np.sum(np.square(W))
        
    J = (-1/m)*(np.sum(Y*np.log(AL+epsilon)+(1-Y)*np.log(1-AL+epsilon))) + (1/m) * (lambd/2) * reg_cost
    J = np.squeeze(J)
    return J

def linear_backward(dZ, linear_cache, lambd):
    A_prev, W, b = linear_cache
    m = A_prev.shape[1]
    
    dW = (1/m) * np.dot(dZ,A_prev.T) + (lambd/m)*W
    db = (1/m) * np.sum(dZ,axis=1,keepdims=True)
    dA_prev = np.dot(W.T,dZ)
    
    return dA_prev, dW, db

def relu_gradient(Z):
    dZ = np.where(Z > 0, 1, 0.01) 
    return dZ

def sigmoid_gradient(Z):
    dZ = special.expit(Z)*(1-special.expit(Z))
    return dZ

def linear_activation_backward(dA, cache, lambd, A, Y, activation):
    linear_cache, activation_cache = cache
    
    if activation == 'relu':
        dZ = dA * relu_gradient(activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache, lambd)
        
    elif activation == 'sigmoid':
        dZ = A - Y
        dA_prev, dW, db = linear_backward(dZ, linear_cache, lambd)

    return dA_prev, dW, db

def L_model_backward(AL, Y, caches, lambd):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) 
        
    cache_final_layer = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(_, cache_final_layer, lambd, AL, Y, activation='sigmoid')
    
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        grads["dA" + str(l)], grads["dW" + str(l+1)], grads["db" + str(l+1)] = linear_activation_backward(grads['dA' + str(l+1)], current_cache, lambd, _, _, activation='relu')
    
    return grads

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
    return parameters

def Neural_Network_Model(X_train, Y_train, X_dev, Y_dev, layers_dims, learning_rate, num_epoch, mini_batch_size, lambd, k):
    
    mini_batches = create_mini_batches(X_train, Y_train, mini_batch_size) #[(X{1},Y{1}),(X{2},Y{2}),...,(X{n},Y{n})]
    
    costs_train = []
    costs_dev = []
    parameters = initialize_parameters(layers_dims)
    
    AL_dev, caches_dev = forward_propagation(X_dev, parameters)
    J_dev = compute_cost(AL_dev, Y_dev, parameters, 0)
    costs_dev.append(J_dev)
    
    for i in range(num_epoch):
        for mini_batch in mini_batches:
            (minibatch_X, minibatch_Y) = mini_batch 
            AL, caches = forward_propagation(minibatch_X, parameters)
            J_train = compute_cost(AL, minibatch_Y, parameters, lambd)
            grads = L_model_backward(AL, minibatch_Y, caches, lambd)
            parameters = update_parameters(parameters, grads, learning_rate)
        if i % 10 == 0:
            costs_train.append(J_train)
            AL_dev, caches_dev = forward_propagation(X_dev, parameters)
            J_dev = compute_cost(AL_dev, Y_dev, parameters, 0)
            costs_dev.append(J_dev)           
        if i % 100 == 0:
            print ("Cost after epoch %i: %f" %(i, J_train))
            learning_rate = learning_rate * (k**(i/50))
            
    plt.plot(np.squeeze(costs_train),'r')
    plt.plot(np.squeeze(costs_dev),'b')
    plt.ylabel('cost')
    plt.xlabel('epochs (per thirties)')
    plt.show()
    
    return parameters, costs_train, costs_dev
  
  
  
if __name__=='__main__':
  parameters_updated, costs_train, costs_dev = Neural_Network_Model(X_train, Y_train, X_dev, Y_dev, [6400, 50, 10, 1], 0.05, 1000, 64, 0.05, 0.95)