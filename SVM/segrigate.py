#segrigating data according to their classes using their indices
def indices(Y, clval):
    Yindices = []
    for i in range(len(Y)):
        if Y[i] == clval:
            Yindices.append(i)
    return Yindices
def segregate(X, Y, cl):
    cl1 = indices(Y, cl[0])
    cl2 = indices(Y, cl[1])
    X_cl1 = []
    for i in cl1:
        X_cl1.append(X[i])
    X_cl2 = []
    for i in cl2:
        X_cl2.append(X[i])

    return X_cl1, X_cl2
