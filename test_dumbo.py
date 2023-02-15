import numpy as np

import dumbo


def _test(measure):
    np.random.seed(17)  # Make the test stable
    x_m = np.atleast_2d(np.array([0.5]))
    y_m = np.array([measure(x_m[0])])

    for _ in range(10):
        x = dumbo.optimize(x_m, y_m)
        y = measure(x)
        x_m = np.append(x_m, x, axis=0)
        y_m = np.append(y_m, y)

    assert y_m.max() > -0.01, y_m.max()


def test_dumbo_1d():
    _test(lambda x: float(-((x - 0.3333) ** 2)))


def test_dumbo_2d():
    _test(lambda x: float(-((x - 0.3333) ** 2).sum()))
