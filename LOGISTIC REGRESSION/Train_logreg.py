from log_reg_cost_function import costfunction
import numpy
from scipy import optimize
import scipy
#training model
def trainlogreg(X,Y,iter):
    [m,n] = numpy.shape(X)
    initial_theta = numpy.zeros([n,1])
    [cost, grad] = costfunction(initial_theta, X, Y)
    option = scipy.optimize.minimize(fun=costfunction, x0= initial_theta, args=(X,Y), method= 'TNC', jac= grad)
    [theta, cost] = scipy.optimize.fmin(initial_theta,(costfunction(initial_theta,X,Y)),option)


