#importing required libraries
import csv
import numpy as np
import scipy.optimize as opt
import random
from sklearn.model_selection import train_test_split
from plot_data import plotdata
from log_reg_cost_function import costfunction
from log_reg_cost_function import gradient
from predict_function import predict
from sklearn import  metrics
from datetime import datetime
import time
# starting time
start_time = time.time()
#reading input file and converting string data into integer
with open('C:\Ruchita\MSc_Data_Science\Module 3-7072 ML\CW_7072\LS_2.0_output.csv','r') as inputdata:
    reader = csv.reader(inputdata)
    data = [[int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[4]), int(row[5]), int(row[6]), int(row[7]), int(row[8])] for row in reader]
#dividing given dataset into training and test data
    random.shuffle(data)
    train_data, test_data = train_test_split(data, test_size=0.4)
#Dividing data according to their class
cl = [0, 1]
X = [a[0:8] for a in train_data]
Y = [a[-1] for a in train_data]

X_test = [a[0:8] for a in test_data]
Y_test = [a[-1] for a in test_data]
#Calculate cost  gradient
m = len(X)
n = len(X[0])
print (m)
print (n)

X1 = np.ones([m,1])
X2 = np.concatenate((X1, X), axis=1)

X1_test = np.ones([len(X_test), 1])
X2_test = np.concatenate((X1_test, X_test), axis=1)


initial_theta = np.zeros([n+1])
Y = np.array([Y]).T
cost = costfunction(initial_theta,X2,Y)
grad = gradient(initial_theta, X2, Y)

print ('Cost at initial theta (zeros): ', cost)
print ('Gradient at initial theta (zeros): ')
print(grad)

result = opt.fmin_tnc(func=costfunction, x0=initial_theta, fprime=gradient, args=(X2, Y))
theta = result[0]
cost = costfunction(theta, X2, Y)
print ('Cost at theta found by fmin_tnc: ', cost)
print ('Theta: ')
print(theta)

#Prediction
p = predict(theta, X2)
print ('Prediction: ')
print (p)

#Accuracy
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b ==0)) else 0 for (a,b) in zip(p,Y)]
accuracy = (sum(map(int, correct)) / len (correct))
print ('Train Accuracy: {0}%'.format(accuracy * 100))

#Confusion Matrix
Y_predict = predict(theta, X2_test)
conf_metrics = metrics.confusion_matrix(Y_test, Y_predict)
print('Confusion Metrics: ')
print(conf_metrics)
print('Accuracy: ', metrics.accuracy_score(Y_test, Y_predict))
print('Precision: ', metrics.precision_score(Y_test, Y_predict))
print('Recall: ', metrics.recall_score(Y_test, Y_predict))
#printing end time of the model
end_time = time.time()
elapsed_time = end_time - start_time
print('Elapsed time is:',elapsed_time)




