import numpy as np
from numpy import linalg as la


def repair_psd(a, eps=1e-3):
    if a.shape == (1,):
        return np.atleast_1d(eps)
    else:
        min_eigval = min(la.eigvals(a))
        if min_eigval >= 0:
            return a
        else:
            return a - (1+eps)*min_eigval*np.identity(a.shape[0])


def inverse_matrix(a):
    if a.shape == (1,):
        return np.atleast_1d(1. / a)
    else:
        return la.inv(a)
