import numpy as np
def cmdscale(z):
#Classical multidimensional scaling (MDS)
 # Number of points
    n = len(z)
# Centering matrix
    C = np.eye(n) - np.ones((n, n))/n
    B = -C.dot(z**2).dot(C)/2
# Diagonalize
    evals, evecs = np.linalg.eigh(B)
    idx   = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:,idx]

    # Compute the coordinates using positive-eigenvalued components onl
    w, = np.where(evals > 0)
    L  = np.diag(np.sqrt(evals[w]))
    V  = evecs[:,w]
    Y  = V.dot(L)

    return Y, evals
