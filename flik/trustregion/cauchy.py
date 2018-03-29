"""Cauchy-point method for the subproblem of the trust-region method."""

import numpy as np

from flik.trustregion.sub_problem import SubProblem

__all__ = ['Cauchy']


class Cauchy(SubProblem):
    """The Cauchy-point method.

    Attributes
    ----------
    point: np.ndarray (1-dimensional)
        Initial point of the optimization.
    func: callable
        Objective function.
    grad: callable
        Gradient of the objective function.
    hess: callable
        Hessian of the objective function.
    radius: float
        The trust region radius.
    current_gradient: np.ndarray (1-dimensional)
        The gradient evaluated at current point.
    current_hessian: np.ndarray (1-dimensional)
        The hessian evaluated at current point.

    Methods
    -------
    __init__(self, point, func, grad, hess, radius, *params)
        Initialize the data corresponding to the given point.
    get_scaling(self)
        Compute the scaling factor.
    solver(self)
        Subproblem Dogleg solver.

    """

    def __init__(self, point, func, grad, hess, radius):
        """Initialize the method.

        Parameters
        ----------
        point: np.ndarray((N,))
            Initial point of the optimization.
        func: callable
            Objective function.
        grad: callable
            Gradient of the objective function.
        hess: callable
            Hessian of the objective function.
        radius: float
            The trust region radius.

        """
        super().__init__(point, func, grad, hess, radius)
        self.current_gradient = self.grad(self.point)
        self.current_hessian = self.hess(self.point)

    def get_scaling(self):
        r"""Define the scaling-factor of the step.

        :math:`\tau = 1` if :math:`g^T B g \leq 0`
        :math:`\tau = \min(||g||^3/(\Delta g^T B g), 1)` otherwise.

        where :math:`g` is the gradient, :math:`B` the Hessian, and
        :math:`\Delta` the trust-region radius.

        Returns:
        --------
        tau: float
            The scaling factor.

        """
        gbg = np.dot(self.current_gradient.T, np.dot(self.current_hessian, self.current_gradient))
        if gbg <= 0.:
            tau = 1.0
        else:
            gnorm_cubic = np.power(np.linalg.norm(self.current_gradient), 3.0)
            tau = min(gnorm_cubic/(self.radius*gbg), 1.0)
        return tau

    def solver(self):
        """Solve the sub-problem using the Cauchy-point.

        Returns
        -------
        step: np.ndarray((N,))
            Correction added to the current point.

        """
        tau = self.get_scaling()
        step = - tau * (self.radius/np.linalg.norm(self.current_gradient)) * self.current_gradient
        return step
