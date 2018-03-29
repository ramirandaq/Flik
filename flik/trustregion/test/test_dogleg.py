"""Tests for the Dogleg method."""

import numpy as np
from numpy.testing import assert_raises
from flik.trustregion.dogleg import Dogleg, get_tau


def test_get_tau():
    """Test tau has the right value."""
    qterm = [1]
    lterm = 0.5
    cterm = -3.0
    # Check parameters
    assert_raises(TypeError, get_tau, qterm, lterm, cterm)
    qterm = 1.0
    lterm = 2
    assert_raises(TypeError, get_tau, qterm, lterm, cterm)
    lterm = 2.0
    cterm = np.array(3)
    assert_raises(TypeError, get_tau, qterm, lterm, cterm)
    cterm = 5.0
    # Check tau value
    assert_raises(ValueError, get_tau, qterm, lterm, cterm)
    # Check actual value of tau
    qterm = 1.0
    lterm = 0.5
    cterm = -3.0
    # Tau not in range (0., 2.)
    assert_raises(ValueError, get_tau, qterm, lterm, -6.0)
    # Right tau
    assert get_tau(1.0, 0.5, -3.0) == 1.5


def fun(x):
    """Test function."""
    return x**3.0


def grad(x):
    """Gradient of the test function."""
    return 3.0 * x**2.0


def hess(x):
    """Hessian of the test function modified to be positive."""
    return np.array([[6.0*x[0], 0.], [0., -6.0*x[1]]])


def test_dogleg():
    """Test Dogleg method."""
    point = np.array([1.5, -0.4])
    radius = 0.7
    # Initialize class
    # Check hessian is positive-definite

    def neg_hess(x):
        """Negative-definite Hessian."""
        return np.array([[-6.0*x[0], 0.], [0., 6.0*x[1]]])

    assert_raises(ValueError, Dogleg, point, fun, grad, neg_hess, radius)
    # Test update_attr
    dogleg = Dogleg(point, fun, grad, hess, radius)
    dogleg.update_attr('point', np.array([1.2, -0.4]))
    assert np.allclose(dogleg.point, np.array([1.2, -0.4]), 1.0e-6)
    assert np.allclose(dogleg.full_step, [-0.6, -0.2], 1.0e-6)
    assert np.allclose(dogleg.steepest_step, [-0.604918, -0.0672131], 1.0e-6)
    # Test when radius the first segment (steepest descent step)
    dogleg.update_attr('point', np.array([1.5, -0.4]))
    result = dogleg.solver()
    assert np.allclose(result, [-0.698237, -0.0496524], atol=1.0e-6)
    # Test when Cauchy point is the solution
    dogleg.radius = 0.8
    result = dogleg.solver()
    assert np.allclose(result, [-0.75, -0.2], atol=1.0e-6)
    # Test when step crosses the second segment
    # Test compute tau
    dogleg.update_attr('point', np.array([1.8, -0.1]))
    dogleg.update_attr('radius', 0.901)
    tau = dogleg.compute_tau()
    assert abs(tau - 0.8392824) < 1.0e-5
    # Test compute_dogleg_step
    step = dogleg.compute_dogleg_step(tau)
    assert np.allclose(step, [-0.900001, -0.0424106], atol=1.0e-6)
    # Test whole Method
    step = dogleg.solver()
    assert np.allclose(step, [-0.900001, -0.0424106], atol=1.0e-6)
