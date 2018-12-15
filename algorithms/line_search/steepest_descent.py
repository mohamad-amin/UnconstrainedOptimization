from algorithms.line_search.line_search_utils import *


def steepest_descent(f, g, x0, alpha0, rho, c, eps=1e-5):
    x = x0
    iterations = 0
    while True:
        p = -g(x)
        alpha = backtrack_step_length(f, g, x, alpha0, p, rho, c)
        next_x = x + alpha * p
        iterations += 1
        if np.allclose(f(x), f(next_x), eps):
            break
        x = next_x
    return next_x, f(next_x), iterations

# x, fx, iterations = steepest_descent(f1, g1, x1, .2, .95, .1, eps=1e-10)

# print('Result in %d iterations:' % iterations)
# print('x -> ', x)
# print('f(x) -> ', fx)
