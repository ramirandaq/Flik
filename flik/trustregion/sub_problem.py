"""Base class for the sub-problem solvers."""
import abc


class SubProblem(abc.ABC):
    """Provide the functionality common to the sub-problem solvers.

    Attributes
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

    Methods
    -------
    __init__(self, point, function, grad, hessian, radius, *params)
        Initialize the data corresponding to the given point.
    solver(self)
        Sub-problem solver.

    """

    def __init__(self, point, func, grad, hess, radius):
        """Initialize the data corresponding to the given point.

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
        self.point = point
        self.func = func
        self.grad = grad
        self.hess = hess
        self.radius = radius

    @abc.abstractmethod
    def solver(self):
        """Sub-problem solver.

        Returns
        -------
        step: np.ndarray((N,))
            Correction added to the current point.

        """
        pass
