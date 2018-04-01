import numpy as np
import argparse

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_(x):
    return x*(1-x)

def tanh(x):
    return np.tanh(x)

def tanh_(x):
    return 1.0 - np.tanh(x)**2

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Choose activation function")
    parser.add_argument('--activation', type=str, default = 'sigmoid', help = 'choose between sigmoid/tanh')

    #Available options but not used like that yet.
    args = parser.parse_args()
    print args.activation
    
    
    X = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
    y = np.array([[0,0,1,1]]).T
    
    np.random.seed(1)
    
    w0 = 2*np.random.random((3,5)) - 1
    w1 = 2*np.random.random((5,1)) - 1
    
    for j in xrange(10000):
        l0=X
        l1=sigmoid(np.dot(X,w0))
        l2=tanh(np.dot(l1,w1))
        l2_error = y - l2
        l2_delta = l2_error * tanh_(l2)
        l1_error = l2_delta.dot(w1.T)
        l1_delta = l1_error * sigmoid_(l1)
    
        w1 += np.dot(l1.T, l2_delta)
        w0 += np.dot(l0.T, l1_delta)
    
    #debug
    #    if j%100 == 0:
    #        print "l1 after iteration" + str(j)
    #        print l1
    #        print "l1 error after iteration" + str(j)
    #        print l1_error
    
    print "Output after training"
    print l2
    
    #print l1
    #print l1_error
