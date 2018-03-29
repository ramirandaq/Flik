"""Dogleg method for the subproblem of the trust-region method."""

import numpy as np

from flik.trustregion.sub_problem import SubProblem


__all__ = ['Dogleg', 'get_tau']


class Dogleg(SubProblem):
    r"""The Dogleg method.

    Attributes
    ----------
    point: np.ndarray (1-dimensional)
        Initial point of the optimization.
    func: callable
        Objective function.
    grad: callable
        Gradient of the objective function.
    hessian: callable
        Hessian of the objective function.
    radius: float
        The trust region radius.
    full_step: np.ndarray (1-dimensional)
        Full step :math:`p^B = - B^{-1} g`.
    steepest_step: np.ndarray (1-dimensional)
        Steepest descent direction step.

    Methods
    -------
    __init__(self, point, function, grad, hessian, radius, *params)
        Initialize the data corresponding to the given point.
    initialize_attributes(self)
        Initialize internal attributes.
    update_attr(self, attr, value)
        Update one attribute and then all the attributes in the class.
    compute_tau(self)
        Solve for the segment-division parameter.
    compute_dogleg_step(self, tau)
        Compute the mixed segment step.
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
        self.initialize_attributes()

    def initialize_attributes(self):
        """Initialize internal attributes."""
        current_gradient = self.grad(self.point)
        current_hessian = self.hess(self.point)
        # Check Hessian is positive definite
        evals = np.linalg.eigvalsh(current_hessian)
        if (evals < 0.).any():
            raise ValueError("The Hessian matrix must be positive-definite")
        # Compute full step
        hessian_inv = np.linalg.inv(current_hessian)
        self.full_step = - np.dot(hessian_inv, current_gradient)
        # Compute steepest-descent direction step
        grad_squared = np.dot(current_gradient.T, current_gradient)
        transformed_hessian = np.dot(current_gradient.T, np.dot(current_hessian, current_gradient))
        self.steepest_step = - (grad_squared / transformed_hessian) * current_gradient

    def update_attr(self, attr, value):
        """Update all attributes after one of the changes.

        Arguments:
        ----------
        attr: str
            Attribute name.
        value:
            New value of the attribute.

        """
        setattr(self, attr, value)
        if attr in ['fun', 'grad', 'hess', 'point']:
            self.initialize_attributes()

    def compute_tau(self):
        r"""Obtain the division parameter :math:`\tau`.

        Solve the quadratic equation:
        :math:`\tau^2 ||p^B-p^U||^2 + 2 \tau (p^B-p^U)p^B + ||p^B||^2 - \Delta^2 = 0`
        """
        # Define polynomial coefficients
        step_diff = self.full_step - self.steepest_step
        qterm = np.dot(step_diff, step_diff)
        lterm = 2.0 * np.dot(self.steepest_step, step_diff)
        steepest_squared = np.dot(self.steepest_step, self.steepest_step)
        cterm = steepest_squared - self.radius**2.0
        tau = get_tau(qterm, lterm, cterm)
        return tau

    def compute_dogleg_step(self, tau):
        r"""Compute the two-line segments step.

        :math:`\tilde{p}(\tau) = \tau p^u if 0 \leq \tau \leq 1`
        :math:`\tilde{p}(\tau) = p^U + (\tau+1)(p^B - p^U)` otherwise

        Arguments:
        ----------
        tau: float
            Line-segments division parameter

        """
        step = self.steepest_step + tau * (self.full_step - self.steepest_step)
        return step

    def solver(self):
        """Find new step.

        Returns
        -------
        step: np.ndarray((N,))
            Correction added to the current point.

        """
        if np.linalg.norm(self.steepest_step) >= self.radius:
            step = (self.radius/np.linalg.norm(self.steepest_step))*self.steepest_step
        elif np.linalg.norm(self.full_step) <= self.radius:
            step = self.full_step
        else:
            tau = self.compute_tau()
            step = self.compute_dogleg_step(tau)
        return step


def get_tau(qterm, lterm, cterm):
    r"""Compute the value of :math:`\tau`.

    Solve the quadratic equation with numpy.roots
    :math:`a x^2 + b x + c = 0`

    Parameters:
    ----------
    qterm: float
        Coefficient of the quadratic term.
    lterm: float
        Coefficient of the linear term.
    cterm: float
        Constant term.

    Returns:
    --------
    tau: float
        Reasonable solution of the quadratic equation

    """
    if not isinstance(qterm, float):
        raise TypeError("Coefficient qterm should be a float.")
    if not isinstance(lterm, float):
        raise TypeError("Coefficient lterm should be a float.")
    if not isinstance(cterm, float):
        raise TypeError("Coefficient cterm should be a float.")
    roots = np.roots([qterm, lterm, cterm])
    if np.iscomplex(roots).all():
        raise ValueError("The roots of the quadratic equation are complex")
    # Use the root with reasonable value
    loc = np.where(roots >= 0.0)
    if roots[loc][0] <= 2.0:
        tau = roots[loc][0]
    else:
        raise ValueError("Neither value of tau is in the right range.")
    return tau
