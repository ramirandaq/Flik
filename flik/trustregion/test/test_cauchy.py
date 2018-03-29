"""Tests for the Cauchy point method."""

import numpy as np
from flik.trustregion.cauchy import Cauchy


def fun(x):
    """Test function."""
    return x**3.0


def grad(x):
    """Gradient of the test function."""
    return 3.0 * x**2.0


def hess(x):
    """Hessian of the test function."""
    return np.array([[6.0*x[0], 0.], [0., 6.0*x[1]]])


def test_cauchy():
    """Test the Cauchy point method."""
    point = np.array([-0.2, -0.1])
    radius = 0.7
    # Initialize class
    cauchy = Cauchy(point, fun, grad, hess, radius)
    # Test get_scaling
    tau = cauchy.get_scaling()
    assert tau == 1.0
    point = np.array([0.2, -0.1])
    cauchy = Cauchy(point, fun, grad, hess, radius)
    tau = cauchy.get_scaling()
    assert abs(tau - 0.161504) < 1.0e-6
    # Test the method
    result = cauchy.solver()
    assert np.allclose(result, [-0.109677, -0.0274193], 1.0e-5)
