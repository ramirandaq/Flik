"""Conditions for the line search algorithm"""

import numpy as np


def strong_wolfe(grad, current_point, current_step, alpha, c_2=0.9):
    """Checks the validity of the strong Wolfe condition

    Parameters
    ----------
    grad: callable
        Gradient of the objective function
    current_point: np.array, float, int
        Current iteration point
        If a float or int are given they are converted to a np.array
    current_step: np.array, float, int
        Current direction along which the minimum will be searched
        If a float or int are given they are converted to a np.array
    alpha: float
        Step length
    c_2: float
        Strong Wolfe condition parameter

    Raises
    -------
    TypeError:
        If grad is not a callable object
        If current_point and current_step are not np.arrays
        If the elements of current_point and current_step are not floats
        If alpha and c_2 are not floats
    ValueError:
        If current_point and current_step have different sizes
        If alpha is not in the interval (0,1]
        If c_2 is not in the interval (0,1)

    Returns
    -------
    bool:
        True if the condition is satisfied, False otherwise
    """

    if not callable(grad):
        raise TypeError("The gradient should be a function")
    if isinstance(current_point, float) or isinstance(current_point, int):
        current_point = np.array(float(current_point))
    elif not isinstance(current_point, np.ndarray):
        raise TypeError("Current point should be a numpy array, float, or int")
    if isinstance(current_step, float) or isinstance(current_step, int):
        current_step = np.array(float(current_step))
    elif not isinstance(current_step, np.ndarray):
        raise TypeError("Current step should be a numpy array, float, or int")
    if current_point.dtype != float:
        raise TypeError("Current point should be given as a numpy array of floats")
    if current_step.dtype != float:
        raise TypeError("Current step should be given as a numpy array of floats")
    if not isinstance(alpha, float):
        raise TypeError("Alpha should be a float")
    if not isinstance(c_2, float):
        raise TypeError("c_2 should be a float")
    if current_point.size != current_step.size:
        raise ValueError("Current point and current step should ve vectors of the same size")
    if not 0 < alpha <= 1:
        raise ValueError("Alpha should be in the interval (0,1]")
    if not 0 < c_2 < 1:
        raise ValueError("c_2 should be in the interval (0,1)")

    return abs(current_step.dot(grad(current_point+alpha*current_step))) <= \
           c_2*abs(current_step.dot(grad(current_point)))