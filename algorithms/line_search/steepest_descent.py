from algorithms.line_search.line_search_utils import *
from functions import *


def update_initial_alpha(f, g, p_x, c_x):
    return 2 * ((f(c_x) - f(p_x)) / np.matmul(g(c_x), -g(c_x)))


def steepest_descent(f, g, x0, alpha0, rho, c, backtrack_selection=True, eps=1e-5):

    x = x0
    iterations = 0
    initial_alpha = alpha0

    while True:

        p = -g(x)
        if backtrack_selection:
            alpha = backtrack_step_length(f, g, x, initial_alpha, p, rho, c)
        else:
            alpha = interpolation_step_length(f, g, x, initial_alpha, p, c)

        next_x = x + alpha * p
        initial_alpha = update_initial_alpha(f, g, x, next_x)

        iterations += 1
        if np.allclose(f(x), f(next_x), eps):
            break

        x = next_x

    return next_x, f(next_x), iterations


# x, fx, iterations = steepest_descent(f1, g1, x1s[0], 1., .9, .01, backtrack_selection=False, eps=1e-12)
#
# print('Result in %d iterations:' % iterations)
# print('x -> ', x)
# print('f(x) -> ', fx)
