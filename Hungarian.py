# Solve the Assignment problem using the Hungarian method.
# input A - square cost matrix
# return - the optimal assignment

import numpy as np
from scipy.optimize import linear_sum_assignment


def Hungarian(A):
    _, col_ind = linear_sum_assignment(A)
    # Cost can be found as A[row_ind, col_ind].sum()
    return col_ind


if __name__ == "__main__":
    pass
