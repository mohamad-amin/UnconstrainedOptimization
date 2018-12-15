from algorithms.linalg_utils import *


def model(f, g, b, x, p, delta):
    if la.norm(p) > delta:
        raise ArithmeticError('P must not be bigger than delta, p and delta:', p, la.norm(p), delta)
    return f(x) + np.matmul(g(x).T, p) + .5*np.matmul(np.atleast_1d(np.matmul(p.T, b)), p)


def trust_region(f, g, hf, x0, delta_0, max_delta, etha, step_finder, repair_hessian=True, eps=1e-5):

    x = x0
    delta = delta_0
    iterations = 0

    while True:

        iterations += 1
        b = hf(x)
        if repair_hessian:
            b = repair_psd(b)

        p = step_finder(g(x), b, delta)
        rho = (f(x) - f(x+p)).astype('f') / (model(f, g, b, x, p, delta) - model(f, g, b, x+p, p, delta))

        if rho < .25:
            delta = .25 * delta
        elif rho >= .75 and np.isclose(la.norm(p), delta, 1e-4):
            delta = min(2*delta, max_delta)

        if rho > etha:
            x = x + p
        elif np.allclose(p, np.zeros(p.shape), eps):
            result = x + p
            break

    return result, f(result), iterations


# x, fx, iterations = trust_region(f2, g2, h2, x2, .1, 1, .15, use_dogleg=False, repair_hessian=True)
#
# print('Result after %d iterations:' % iterations)
# print('x -> ', x)
# print('f(x) -> ', fx)
