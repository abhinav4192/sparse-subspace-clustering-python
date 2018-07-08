# This function takes the D x N data matrix with columns indicating
# different data points and project the D dimensional data into the r
# dimensional space. Different types of projections are possible:
# (1) Projection using PCA
# (2) Projection using random projections with iid elements from N(0,1/r)
# (3) Projection using random projections with iid elements from symmetric
# bernoulli distribution: +1/sqrt(r),-1/sqrt(r) elements with same probability
# X: D x N data matrix of N data points
# r: dimension of the space to project the data to
# type: type of projection, {'PCA','NormalProj','BernoulliProj'}
# Xp: r x N data matrix of N projectred data points

import numpy as np
import math


def DataProjection(X, r, type='NormalProj'):
    Xp = None
    D, N = X.shape
    if r == 0:
        Xp = X
    else:
        if type == 'PCA':
            isEcon = False
            if D > N:
                isEcon = True
            U, S, V = np.linalg.svd(X.T, full_matrices=isEcon)
            Xp = U[:, 0:r].T
        if type == 'NormalProj':
            normP = (1.0 / math.sqrt(r)) * np.random.randn(r * D, 1)
            PrN = normP.reshape(r, D, order='F')
            Xp = np.matmul(PrN, X)
        if type == 'BernoulliProj':
            bp = np.random.rand(r * D, 1)
            Bp = (1.0 / math.sqrt(r)) * (bp >= .5) - (1.0 / math.sqrt(r)) * (bp < .5)
            PrB = Bp.reshape(r, D, order='F')
            Xp = np.matmul(PrB, X)
    return Xp


if __name__ == "__main__":
    pass
