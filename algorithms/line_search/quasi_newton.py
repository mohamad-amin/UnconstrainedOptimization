from algorithms.linalg_utils import *
from line_search_utils import *


class HessianMethod:
    HESSIAN = 0
    BFGS = 1
    SR1 = 2


def approximate_hessian_sr1(g, x, next_x, last_h):
    s = next_x - x
    y = g(next_x) - g(x)
    helper = s - np.matmul(last_h, y)
    h = last_h + (np.matmul(helper, helper.T).astype('f') / np.matmul(helper.T, y))
    return h


def approximate_hessian_bfgs(g, x, next_x, last_h):
    s = next_x - x
    y = g(next_x) - g(x)
    rho = 1. / np.matmul(y.T, s)
    helper = np.identity(x.shape[0]) - rho*np.matmul(s, y.T)
    h = helper*last_h*helper + rho*np.matmul(s, s.T)
    return h


def quasi_newton(f, g, hf, x0, c, r, alpha0, backtrack_selection=True, repair_hessian=True,
                 use_identity_h0=False, hessian_method=HessianMethod.HESSIAN, eps=1e-5):

    iterations = 0
    x = x0

    if use_identity_h0:
        h = np.identity(x.shape[0])
    else:
        h = inverse_matrix(hf(x))
    if repair_hessian:
        h = repair_psd(h)

    while True:

        iterations += 1
        p = -np.atleast_1d(np.matmul(h, g(x)))

        if backtrack_selection:
            alpha = backtrack_step_length(f, g, x, alpha0, p, r, c)
        else:
            alpha = interpolation_step_length(f, g, x, alpha0, p, c)

        next_x = x + alpha*p
        if np.allclose(f(x), f(next_x), eps):
            break

        if hessian_method == HessianMethod.HESSIAN:
            h = inverse_matrix(hf(next_x))
        elif hessian_method == HessianMethod.BFGS:
            h = approximate_hessian_bfgs(g, x, next_x, h)
        else:
            h = approximate_hessian_sr1(g, x, next_x, h)
        if repair_hessian:
            h = repair_psd(h)

        x = next_x

    return next_x, f(next_x), iterations


# x, fx, iterations = quasi_newton(f2, g2, h2, x2, .01, 0.9, .1, repair_hessian=False,
#                                  backtrack_selection=True, hessian_method=HessianMethod.SR1)
#
# print('Result after %d iterations:' % iterations)
# print('x -> ', x)
# print('f(x) -> ', fx)

