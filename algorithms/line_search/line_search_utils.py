import numpy as np


def step_has_sufficient_decrease(f, g, x, alpha, p, c):
    return f(x + alpha*p) <= f(x) + c * alpha * np.matmul(g(x).T, p)


def step_has_curvature_condition(g, x, alpha, p, c):
    return np.matmul(g(x + alpha*p).T, p) >= c * np.matmul(g(x).T, p)


def step_satisfies_wolfe_conditions(f, g, x, alpha, p, c1, c2):
    return step_has_sufficient_decrease(f, g, x, alpha, p, c1) \
           and step_has_curvature_condition(g, x, alpha, p, c2) \
           and c2 < c1


def step_satisfies_goldstein_condition(f, g, x, alpha, p, c):
    result = f(x + alpha*p)
    first_cond = f(x) + (1-c)*alpha*np.matmul(g(x).T, p) <= result
    second_cond = result <= f(x) + c*alpha*np.matmul(g(x).T, p)
    return first_cond and second_cond


def backtrack_step_length(f, g, x, alpha, p, r, c):
    length = alpha
    while not step_has_sufficient_decrease(f, g, x, length, p, c):
        length = r * length
    return length


def interpolation_step_length(f, g, x, alpha0, p, c):
    if step_has_sufficient_decrease(f, g, x, alpha0, p, c):
        return alpha0
    alpha = alpha0
    phi_0 = f(x)
    phi_prime_0 = np.matmul(g(x).T, p)
    phi_alpha_0 = f(x + alpha*p)
    next_alpha = quadratic_interpolation_step_length(phi_0, phi_prime_0, phi_alpha_0, alpha)
    while not step_has_sufficient_decrease(f, g, x, next_alpha, p, c):
        phi_alpha = f(x + alpha*p)
        phi_next_alpha = f(x + alpha*next_alpha)
        new_alpha = cubic_interpolation_step_length(alpha, next_alpha, phi_alpha, phi_next_alpha, phi_0, phi_prime_0)
        if new_alpha <= next_alpha/4. or np.allclose(new_alpha, next_alpha, 5e-2) or np.isnan(new_alpha):
            alpha = next_alpha
            next_alpha = next_alpha/2
        else:
            alpha = next_alpha
            next_alpha = new_alpha
    return next_alpha


def quadratic_interpolation_step_length(phi_0, phi_prime_0, phi_alpha, alpha):
    return - (phi_prime_0 * alpha**2) / (2.*(phi_alpha - phi_0 - phi_prime_0*alpha))


def cubic_interpolation_step_length(alpha_0, alpha_1, phi_alpha_0, phi_alpha_1, phi_0, phi_prime_0):
    helper1 = 1. / alpha_0**2 * alpha_1**2 * (alpha_1 - alpha_0)
    helper2 = np.array([[alpha_0**2, -(alpha_1**2)], [-alpha_0**3, alpha_1**3]])
    helper3 = np.array([phi_alpha_1 - phi_0 - alpha_1*phi_prime_0, phi_alpha_0 - phi_0 - alpha_0*phi_prime_0])
    helper = helper1 * np.matmul(helper2, helper3)
    a = helper[0]
    b = helper[1]
    s = b**2 - 3*a*phi_prime_0
    if np.isnan(s) or s < 0:
        return np.nan
    return (-b + np.sqrt(s)) / (3*a)
