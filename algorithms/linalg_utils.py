import numpy as np
from numpy import linalg as la


def is_psd(a):
    if a.shape == (1,):
        return a >= 0
    else:
        return np.all(la.eigvals(a) >= 0)


def repair_psd(a, eps=1e-3):
    if is_psd(a):
        return a
    if a.shape == (1,):
        return np.atleast_1d(eps)
    else:
        min_eigval = min(la.eigvals(a))
        if min_eigval >= 0:
            return a
        else:
            x = a - (1+eps)*min_eigval*np.identity(a.shape[0])
            return x


def inverse_matrix(a):
    if a.shape == (1,):
        return np.atleast_1d(1. / a)
    else:
        return la.inv(a)
