from algorithms.linalg_utils import *


def cauchy_point_step_finder(gx, b, delta):
    gt_b_g = np.matmul(np.atleast_1d(np.matmul(gx.T, b)), gx)
    g_norm = la.norm(gx)
    if gt_b_g <= 0:
        taw = 1
    else:
        taw = min(g_norm**3./(delta*gt_b_g), 1.)
    cp = -1. * (taw*delta/g_norm) * gx
    mul = np.floor(delta / la.norm(cp).astype('f')) * (1-1e-3)
    if mul == 0:
        return cp * (1-1e-3)
    else:
        return mul * cp


def solve_taw_for_dogleg(pu, pb, delta):
    a = np.dot(pu-pb, pu-pb)
    b = -2 * (2*np.dot(pu, pu) + np.dot(pb, pb) - 3*np.dot(pu, pb))
    c = np.dot(2*pu-pb, 2*pu-pb) - delta**2
    d = np.sqrt(b**2 - 4*a*c)
    t1 = (-b + d) / (2*a)
    t2 = (-b - d) / (2*a)
    if 0 <= t1 <= 2:
        if 0 <= t2 <= 2:
            return min(t1, t2)
        return t1
    elif 0 <= t2 <= 2:
        return t2
    else:
        raise ArithmeticError('Taw is not in [0,2]: %d %d', t1, t2)


def dogleg_step_finder(gx, b, delta):
    pb = -np.atleast_1d(np.matmul(inverse_matrix(b), gx))
    if la.norm(pb) <= delta:
        return pb
    pu = - (np.matmul(gx.T, gx).astype('f') / (np.matmul(np.atleast_1d(np.matmul(gx.T, b)), gx))) * gx
    if la.norm(pu) >= delta:
        return (delta / la.norm(pu).astype('f')) * (1-1e-3) * pu
    taw = solve_taw_for_dogleg(pu, pb, delta)
    if taw <= 1:
        return taw * pu
    else:
        return pu + (taw - 1)*(pb - pu)
