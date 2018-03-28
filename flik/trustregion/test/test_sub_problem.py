"""Tests for flik.trustregion.sub_problem."""
import numpy as np
from flik.trustregion.sub_problem import SubProblem


def test_sub_problem():
    """Test the SubProblem base class."""
    # Set parameters
    point = np.array([1., 2.])

    def func(point):
        """Objective function."""
        return point**3

    def grad(point):
        """Gradient of the objective function."""
        return 3*point**2

    def hess(point):
        """Hessian of the objective function."""
        return np.array([[6*point[0], 0.], [0., 6*point[1]]])
    radius = 0.9

    class EmptySubProblem(SubProblem):
        """Class with no solver method."""

        pass

    try:
        EmptySubProblem(point, func, grad, hess, radius)
    except TypeError:
        pass
        # "Child class does not have a solver method"

    class SimpleSubProblem(SubProblem):
        """Class to test the SubProblem base class."""

        def solver(self):
            """Sub-problem solver."""
            hessian = self.hess(self.point)
            gradient = self.grad(self.point)
            return -np.linalg.inv(hessian).dot(gradient)

    a = SimpleSubProblem(point, func, grad, hess, radius)
    assert np.array_equal(a.point, np.array([1., 2.]))
    assert np.array_equal(a.func(a.point), np.array([1., 8.]))
    assert np.array_equal(a.grad(a.point), np.array([3., 12.]))
    assert np.array_equal(a.hess(a.point), np.array([[6., 0.], [0., 12.]]))
    assert np.allclose(a.solver(), -np.array([0.5, 1.]))
