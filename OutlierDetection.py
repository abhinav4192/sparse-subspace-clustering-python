# This function takes the coefficient matrix resulted from sparse
# representation using \ell_1 minimization. If a point cannot be written as
# a linear combination of other points, it should be an outlier. The
# function detects the indices of outliers and modifies the coefficient
# matrix and the ground-truth accordingly.
# CMat: NxN coefficient matrix
# s: Nx1 ground-truth vector
# CMatC: coefficient matrix after eliminating Nans
# sc: ground-truth after eliminating outliers
# OutlierIndx: indices of outliers in {1,2,...,N}
# Fail: True if number of inliers is less than number of groups, False otherwise

import numpy as np


def OutlierDetection(CMat, s):
    n = np.amax(s)
    _, N = CMat.shape
    OutlierIndx = list()
    FailCnt = 0
    Fail = False

    for i in range(0, N):
        c = CMat[:, i]
        if np.sum(np.isnan(c)) >= 1:
            OutlierIndx.append(i)
            FailCnt += 1
    sc = s.astype(float)
    sc[OutlierIndx] = np.nan
    CMatC = CMat.astype(float)
    CMatC[OutlierIndx, :] = np.nan
    CMatC[:, OutlierIndx] = np.nan
    OutlierIndx = OutlierIndx

    if FailCnt > (N - n):
        CMatC = np.nan
        sc = np.nan
        Fail = True
    return CMatC, sc, OutlierIndx, Fail


if __name__ == "__main__":
    pass
