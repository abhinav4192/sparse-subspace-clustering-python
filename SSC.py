import numpy as np
from DataProjection import *
from BuildAdjacency import *
from OutlierDetection import *
from BestMap import *
from SpectralClustering import *
from SparseCoefRecovery import *


def SSC_test():
    # Basic test to check SSC.

    D = 30  # Dimension of ambient space
    n = 2  # Number of subspaces
    d1 = 1
    d2 = 1  # d1 and d2: dimension of subspace 1 and 2
    N1 = 50
    N2 = 50  # N1 and N2: number of points in subspace 1 and 2
    # Generating N1 points in a d1 dim. subspace
    X1 = np.random.randn(D, d1) * np.random.randn(d1, N1)
    # Generating N2 points in a d2 dim. subspace
    X2 = np.random.randn(D, d2) * np.random.randn(d2, N2)
    X = np.concatenate((X1, X2), axis=1)

    # Generating the ground-truth for evaluating clustering results
    s = np.concatenate((1 * np.ones([1, N1]), 2 * np.ones([1, N2])), axis=1)
    r = 0  # Enter the projection dimension e.g. r = d*n, enter r = 0 to not project
    Cst = 0  # Enter 1 to use the additional affine constraint sum(c) == 1
    OptM = 'L1Perfect'  # OptM can be {'L1Perfect','L1Noise','Lasso','L1ED'}
    lmbda = 0.001  # Regularization parameter in 'Lasso' or the noise level for 'L1Noise'
    # Number of top coefficients to build the similarity graph, enter K=0 for using the whole coefficients
    K = max(d1, d2)
    if Cst == 1:
        K = max(d1, d2) + 1  # For affine subspaces, the number of coefficients to pick is dimension + 1

    Xp = DataProjection(X, r, 'NormalProj')
    CMat = SparseCoefRecovery(Xp, Cst, OptM, lmbda)
    # Make small values 0
    eps = np.finfo(float).eps
    CMat[np.abs(CMat) < eps] = 0

    CMatC, sc, OutlierIndx, Fail = OutlierDetection(CMat, s)

    if Fail == False:
        CKSym = BuildAdjacency(CMatC, K)
        Grps = SpectralClustering(CKSym, n)
        Grps = BestMap(sc, Grps)
        Missrate = float(np.sum(sc != Grps)) / sc.size
        print("Misclassification rate: {:.4f} %".format(Missrate * 100))
    else:
        print("Something failed")


if __name__ == "__main__":
    SSC_test()
