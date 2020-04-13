#importing required libraries
import csv
from sklearn.model_selection import train_test_split,cross_val_score
import random
from scipy.spatial import distance
from scipy.spatial.distance import squareform
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from cmdscale import cmdscale
import itertools
from segrigate import segregate
from sklearn import metrics
import time
#Calculating Starting time
start_time = time.time()
#Reading data and converting features in integers from string
with open('C:\Ruchita\MSc_Data_Science\Module 3-7072 ML\CW_7072\LS_2.0_output.csv','r') as inputdata:
#shuffling data and deviding it into training and test data
    reader = csv.reader(inputdata)
    data =[[int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[4]), int(row[5]), int(row[6]), int(row[7]), int(row[8])] for row in reader]

    random.shuffle(data)
    data_train,data_test = train_test_split(data, test_size=0.4)
data_train = np.array(data_train)
data_test = np.array(data_test)
cl = [0,1]
#Calculating Euclidian for all possible combinations and converting it to sqare form matrix
print(list(itertools.combinations(range(len(data_train)),2)))
D = distance.pdist(data_train[:,0:8],'euclidean')
print('D',D)
z = squareform (D)
#Applying classical multidimensional scaling
[Y,e] = cmdscale(z)
#Dividing Y class wise
Y_0_cl1, Y_0_cl2 = segregate(Y[:,0], data_train[:,-1], cl)
Y_1_cl1,Y_1_cl2 = segregate(Y[:,1],data_train[:,-1],cl)
#Plotting scatterd graph
plt.plot(Y_0_cl1,Y_1_cl1,'.',color = 'blue')
plt.plot(Y_0_cl2,Y_1_cl2,'v',color = 'red')
plt.show()
#Training SVM model using sklearn library
SVMModel = svm.SVC(kernel= 'sigmoid', C = 1.0)
SVMModel.fit(data_train[:,0:8],data_train[:,-1])
#cross validation classification
CVSVMModel = cross_val_score(SVMModel, data_train[:,0:8], data_train[:,-1], cv=5)
#print predicted values of classes
print(SVMModel.predict(data_test[:,0:8]))
#Printing Accuracy and confusion matrix
print("Accuracy: {0:.4f}".format(metrics.accuracy_score(data_test[:,-1], SVMModel.predict(data_test[:,0:8]))))
print('confusion matrix:',(metrics.confusion_matrix(data_test[:,-1], SVMModel.predict(data_test[:,0:8]))))
#Calculating elapsed time
end_time = time.time()
elapsed_time = end_time-start_time
print('Elapsed time:',elapsed_time)
