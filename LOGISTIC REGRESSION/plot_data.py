import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')


def indices(Y, clval):
    Yindices = []
    for i in range(len(Y)):
        if Y[i] == clval:
            Yindices.append(i)
    return Yindices


def plotdata(X, Y, cl):
    cl1 = indices(Y, cl[0])
    cl2 = indices(Y, cl[1])

    XaxisdataCl1 = []
    for i in cl1:
        XaxisdataCl1.append(X[i][2])

    YaxisdataCl1 = []
    for i in cl1:
        YaxisdataCl1.append(X[i][1])

    XaxisdataCl2 = []
    for i in cl2:
        XaxisdataCl2.append(X[i][2])

    YaxisdataCl2 = []
    for i in cl2:
        YaxisdataCl2.append(X[i][1])

    plt.plot(XaxisdataCl1, YaxisdataCl1, 'o', color='red', label = 'Lost')
    plt.plot(XaxisdataCl2, YaxisdataCl2, 'x', color='blue', label = 'Won')
    plt.xlabel('Party')
    plt.ylabel('Constituency')
    plt.title('Indian Loksabha Election results')
    plt.legend()

    plt.show()
    plt.plot()
