import numpy as np

from functions import ScalarFunction


class FiniteDifference(ScalarFunction):
    def __init__(self, h=1e-10): #making h smaller helps converge (initially h=1e-3)
        super(FiniteDifference, self).__init__()
        self.h = h

    def jacobian(self, x):
        x1, x2 = x.squeeze()
        # TO DO: Replace this with the finite difference approximation of the first derivatives.
        # Remember that this should be a vector of partial derivatives (e.g. a 2x1 vector for a 2D function)
        derivative = np.ones((2, 1));
        e1 = np.array([[1, 0]]).T;
        e2 = np.array([[0, 1]]).T;
        derivative[0,0] = (self.__call__(x + self.h*e1) - self.__call__(x))/self.h;
        derivative[1,0] = (self.__call__(x + self.h*e2) - self.__call__(x))/self.h;
        
        assert derivative.shape == (2, 1), 'jacobian must be a 2x1 vector';
        return derivative;

    def hessian(self, x):
        raise NotImplementedError('finite difference has not been defined for hessian')


def finite_difference(function_type, *args, **kwargs):
    """Create an instance of a function with finite differences instead of analytical derivatives

    Examples:
        For a function that implements derivatives this uses finite differences instead. For a relatively smooth
        function you would expect the value to be comparable to the exact derivative.

        >>> rosenbrock = functions.Rosenbrock()
        >>> type(rosenbrock)
        functions.Rosenbrock
        >>> rosenbrock.jacobian(np.array([[1, 2]]).T)
        array([[-400],
               [ 200]])

        >>> rosenbrock = finite_difference(functions.Rosenbrock)
        >>> type(rosenbrock)
        numerical.RosenbrockFiniteDifference
        >>> rosenbrock.jacobian(np.array([[1, 2]]).T)
        array([[-400.0008006],
               [ 200.       ]])
    """
    name = '{}FiniteDifference'.format(function_type.__name__)
    finite_difference_function = type(name, (FiniteDifference, function_type), {})
    return finite_difference_function(*args, **kwargs)
