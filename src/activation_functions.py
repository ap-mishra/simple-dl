import numpy as np

#sigmoid activation function
def sigmoid(x):
    return 1/(1+np.exp(-x))

#sigmoid derivative
def sigmoid_(x):
    return x*(1-x)

#tanh activation function
def tanh(x):
    return np.tanh(x)

#tanh derivative
def tanh_(x):
    return 1.0 - np.tanh(x)**2

