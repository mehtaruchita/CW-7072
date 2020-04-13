import numpy as np
#sigmoid function
def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    return g
def costfunction(theta, X, Y):
    m = len(Y)
    j = 0
    y1 = np.ones([m, 1])
#Computing cost function and function for gradient
    theta = np.array([theta])
    first = -np.transpose(Y).dot(np.log(sigmoid(X.dot(theta.T))))
    second = np.transpose(y1-Y).dot(np.log(y1-sigmoid(X.dot(theta.T))))
    j = (1/m) * (first-second)
    return j[0]

def gradient(theta, X, Y):
    theta = np.array([theta])
    m = len(Y)
    error = sigmoid(X.dot(theta.T))-Y
    grad = (1/m) * np.transpose(X).dot(error)
    return grad
