import numpy
from log_reg_cost_function import sigmoid

#prediction function
def predict(theta,X):
    p = sigmoid(X.dot(theta))
    return [1 if x >= 0.5 else 0 for x in p]
