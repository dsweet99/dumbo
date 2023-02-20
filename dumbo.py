"""The simplest Bayesian optimizer

   DumBO performs Bayesian optimization using the most simplistic
    version of each essential component of the algorithm:
    - The surrogate is a 1-nearest-neighbor model.
    - The acquisition function is UCB
    - The optimizer is a random search.

   DumBO was built  teaching, but it does
    also kind of work.

   NOTE
    - The bounds on x are [0,1]**num_dim
"""

import numpy as np


def acquisition_function(y_ex, y_se):
    """Upper Confidence Bound (UCB)
    Seek points that have high value (y_ex) but also
     high uncertainty (y_se).
    Put another way: Balance exploration (y_se) with
     exploitation (y_ex).
    Biasing the next measurement towards areas where
     the surrogate is uncertain makes us take a measurement
     that will make the surrogate less uncertain.

    UCB is great for teaching and simple to write out,
     but it has a reputation for over-exploring, so
     we divide the second term by 2. (Hey, this is
     DumBO, not SmartBO!)
    """
    return y_ex + np.sqrt(y_se) / 2


def surrogate(x_m, y_m, x):
    """GPR-1NN
    A simple surrogate.
    As a GPR, each point, y(x), is modeled by a
     gaussian distribution with mean E[y(x)]
     and variance VAR[y(x)].
    As a simple model, all information about y(x)
     is taken from the nearest neighbor.

    E[y(x)] = value of nearest neighbor
    VAR[y(x)] = distance from nearest neighbor
    """
    distance = np.sqrt(((x - x_m) ** 2).sum(axis=1))
    i_nn = np.argmin(distance)
    y_ex = y_m[i_nn]
    y_var = distance[i_nn]
    y_se = np.sqrt(y_var)
    return y_ex, y_se


def _standardize(y):
    return (y - y.mean()) / (1e-9 + y.std())


def optimize(x_m, y_m, eps=0.1, num_iterations=1000, bounds=None):
    """Random search
    Maximize the acquisition function.
    The x value that maximizes the acquisition function
     is the next point to measure.
    Put another way, we design the next experiment
     by maximizing the acquisition function.
    If you don't specify bounds, we'll just make them [0,1] in
     all dimensions.
    """

    y_m_s = _standardize(y_m)
    num_dim = len(x_m[0])
    x_best = np.random.uniform(size=(num_dim,))
    if bounds is None:
        bounds = np.array([[0.0] * num_dim, [1.0] * num_dim])
    af_best = -1e99
    for _ in range(num_iterations):
        x = np.minimum(bounds[1], np.maximum(bounds[0], x_best + eps * np.random.uniform(size=(num_dim,))))
        x = np.atleast_2d(x)
        af = acquisition_function(*surrogate(x_m, y_m_s, x))
        if af > af_best:
            af_best = af
            x_best = x
    return x_best
