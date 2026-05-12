from scipy.optimize import minimize
from numpy.linalg import norm
import numpy as np


def solve_lagrange(x0: tuple[float, float, float]):
    """
    Uses Lagrange Multipliers to find shortest point on ellipse to x0

    returns minimum point and distance pair
    """

    fun = lambda x: (x[0]-x0[0])**2+(x[1]-x0[1])**2+(np.sqrt(1-x[0]**2-x[1]**2)-x0[2])**2
    res = minimize(fun, [0,0])
    return res




def find_min(x, surface):
    """
    Minimizes L2 norm between a fixed point and a surface. x should be an ndarray and surface should be 
    a lambda function that takes one ndarray as input and return 0 if the point is on the surface. 

    returns a scipy.optimize.OptimizeResult object. 

    EXAMPLE USAGE:
        x0 = (0,0,0)
        surface = lambda x: x[0]**2 + x[1]**2 + x[2]**2 - 1 (sphere of radius 1)
        res = find_min(x0, surface)
        print(res.x)
    """
    x = np.array(x)
    cons = {'type': 'eq', 'fun': surface}
    return minimize(distance, x, args=(x), constraints=cons)

def distance(x, *args):
    """
    Returns the norm of a point and a fixed point. Formatted as required by 
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    """
    return norm(x - args[0])



if __name__ == "__main__":
    print(solve_lagrange((3,0.1,2)))