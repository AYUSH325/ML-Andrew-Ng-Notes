from matplotlib import widgets
import numpy as np
# %matplotlib widget
import matplotlib.pyplot as plt
from lab_utils_uni import plt_intuition, plt_stationary, plt_update_onclick, soup_bowl

def compute_cost(x, y, w, b):
    """
    Computes the cost function for linear regression.

    Args:
        x (ndarray (m,)): m examples
        y (ndarray (m,)): target values
        w,b (scalar)    : model parameters

    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
            to fit the data points in x and y
    """

    # number of training examples
    m = x.shape[0]

    cost_sum = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost = (f_wb - y[i]) ** 2
        cost_sum = cost_sum + cost
    total_cost = 1/(2 * m) * cost_sum

    return total_cost

x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])

plt_intuition(x_train, y_train)


