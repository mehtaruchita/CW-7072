#Importing required Libraries
import csv
import random
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from lda_input import segregate
import time
from sklearn.metrics import confusion_matrix
#start calculating time
start_time = time.time()
#input data and convert string into integer
with open('C:\Ruchita\MSc_Data_Science\Module 3-7072 ML\CW_7072\LS_2.0_output.csv', 'r') as inputdata:
    reader = csv.reader(inputdata)
    data = [[int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[4]), int(row[5]), int(row[6]), int(row[7]), int(row[8])] for
            row in reader]
    # Dividing data into training set and test set
    random.shuffle(data)
    train_data, test_data = train_test_split(data, test_size=0.4)
# division of Training data and test data into class_0 and class_1
cl = [0, 1]

X_train = [a[0:8] for a in train_data]
Y_train = [a[-1] for a in train_data]

X_test = [a[0:8] for a in test_data]
Y_test = [a[-1] for a in test_data]
#calling segregate function written in another python file
X_train_cl1, X_train_cl2 = segregate(X_train, Y_train, cl)
print('X_train_cl1 is \n', X_train_cl1)
print('X_train_cl2 is \n', X_train_cl2)

X_test_cl1, X_test_cl2 = segregate(X_test, Y_test, cl)
print('X_test_cl1 is \n', X_test_cl1)
print('X_test_cl2 is \n', X_train_cl2)

#Calculation of Mean and Mu
[N_zero, P] = np.shape(X_train_cl1)
[N_one, P] = np.shape(X_train_cl2)

mu = np.mean(X_train, axis=0)
print('The value for Mu :', mu)

mu_0 = np.mean(X_train_cl1, axis=0)
mu_1 = np.mean(X_train_cl2, axis=0)
mu_k = []
mu_k.append(mu_0)
mu_k.append(mu_1)
# calculation of Posterior Probability
Nc =[]
Nc.append(N_zero)
Nc.append(N_one)

per_0 = N_zero / np.sum(N_zero + N_one)
per_1 = N_one / np.sum(N_zero + N_one)
print('Posterior Probability for 0 is:',per_0)
print('Posterior Probability for 1 is:',per_1)

#Scatter within the class
a = (X_train_cl1 - mu_0)
scatter_loss = numpy.dot((X_train_cl1 - mu_0).T, (X_train_cl1 - mu_0))
scatter_win = numpy.dot((X_train_cl2 - mu_1).T, (X_train_cl2 - mu_1))
SW = scatter_loss + scatter_win
print("Scatter within the class: ")
print(SW)

#Scatter between the class
SB = np.dot(Nc*np.array(mu_k-mu).T,np.array(mu_k-mu))

#Eigen Value and Vector calculation
eigenval, eigenvec = np.linalg.eig(np.dot(np.linalg.inv(SW), SB))

#Identify 2 largest eigen values
eigenpairs = [[np.abs(eigenval[i]),eigenvec[:,i]] for i in range(len(eigenval))]
eigenpairs = sorted(eigenpairs,key=lambda k: k[0],reverse=True)
w = np.hstack((eigenpairs[0][1][:,np.newaxis].real,eigenpairs[1][1][:,np.newaxis].real))

#Transform the data Y = X * w
Y = np.dot(X_train, w)

#plot the data

fig = plt.figure(figsize=(10,10))
ax0 = fig.add_subplot(111)
ax0.set_xlim(-5,14)
ax0.set_ylim(-5,14)
for l,c,m in zip([0,1],['r','g'],['x','o']):
    ax0.scatter(Y.T[0][np.array(Y_train)==l],
                Y.T[1][np.array(Y_train)==l],
               c=c, marker=m, label=l,edgecolors='black')
ax0.legend(loc='upper right')

means = []
for m,target in zip(['x','o'],[0,1]):
    means.append(np.mean(Y[np.array(Y_train)==target],axis=0))
    ax0.scatter(np.mean(Y[np.array(Y_train)==target],axis=0)[0],np.mean(Y[np.array(Y_train)==target],axis=0)[1],marker=m,c='black',s=100)

mesh_x, mesh_y = np.meshgrid(np.linspace(-5,14),np.linspace(-5,14))
mesh = []
for i in range(len(mesh_x)):
    for j in range(len(mesh_x[0])):
        date = [mesh_x[i][j],mesh_y[i][j]]
        mesh.append((mesh_x[i][j],mesh_y[i][j]))
NN = KNeighborsClassifier(n_neighbors=1)
NN.fit(means,['r','g'])
predictions = NN.predict(np.array(mesh))

ax0.scatter(np.array(mesh)[:,0],np.array(mesh)[:,1],color=predictions,alpha=0.3)


plt.show()
#Training model using sklearn LinearDiscriminantAnalysis and Calculating Accuracy(Using score)
lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto',n_components=2)
lda.fit(np.array(X_train), np.array(Y_train))
#X_lda = lda.transform(np.array(X_train))

Y_pred = lda.predict(X_test)
print('Y Predicted: ')
print(Y_pred)
#Calculation of Accuracy
score = lda.score(np.array(X_test), np.array(Y_test))
print('Score: ',score)
#Calculation of confusion matrix
cm = confusion_matrix(np.array(Y_test), np.array(Y_pred))
print('Confusion Matrix: ')
print(cm)
#ending time
end_time = time.time()
elapsed_time = end_time - start_time
print('Elapsed time:',elapsed_time)





