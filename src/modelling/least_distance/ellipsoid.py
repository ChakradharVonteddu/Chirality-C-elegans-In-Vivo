import numpy as np
from . import minimize
import time


def min_point_ellpsoid(y: tuple[float, float, float], e0: float, e1: float) -> np.ndarray:
    """
    Finds distance from a point in R3 to an ellipsoid. Assumes the ellipsoid is 
    a Prolate Spheriod where e0 > e1 = e2 for the ellipsoid with equation 
        x0^2/e0^2 + (x1^2+x2^2)/e1^2 
    https://www.geometrictools.com/Documentation/DistancePointEllipseEllipsoid.pdf


    we let x = x0, y = x1, z = x2 since the x axis separates the anterior and posterior. 
    change of variables r = sqrt(y1^2+y2^2)

    requires e0 >= e1 > 0
    """    

    # on ellipse; return
    if ((y[0]/e0)**2 + (y[0]/e1)**2 + (y[0]/e1)**2 - 1) == 0:
        return np.array(y)

    y0 = abs(y[0])
    y1 = abs(y[1])
    y2 = abs(y[2])

    r = np.sqrt(y[1]**2+y[2]**2) #change of variables to 2 dimensional problem; we interpret the Prolate Spheriod as a 2D ellipse
    sol = [0,0,0]

    if r > 0:
        if y0 > 0:
            t_bar = get_ellipse_root(y0, r, e0, e1)
            sol[0] = (e0**2)*y0 / (t_bar+e0**2)
            radius = (e1**2)*r / (t_bar+e1**2)
        else: #y0 == 0
            radius = e1
    else: # r == 0
        if y0 < ((e0**2 - e1**2)/e0):
            sol[0] = (e0**2)*y0/(e0**2-e1**2)
            radius = e1*np.sqrt(1-(sol[0]/e0)**2) # the case where r = 0 and y0 = e0 is on the ellipse, so is covered already
        else:
            sol[0] = e0
            return np.array(sol)

    # find min (y1-x1)**2+(y2-x2)**2 such that (x1,x2) lie on a circle with rad r
    min_circle_p = radius * np.array([y1, y2]) / np.linalg.norm([y1, y2])
    sol[1] = min_circle_p[0]
    sol[2] = min_circle_p[1]

    # use symmetry to deduce signs
    if y[0] < 0:
        sol[0] *= -1
    if y[1] < 0:
        sol[1] *= -1
    if y[2] < 0:
        sol[2] *= -1

    return np.array(sol)

    

def get_ellipse_root(y0, y1, e0, e1, maxit = 2000):
    """
    Root finding for ellipse using the Bisection Method. 
    https://www.geometrictools.com/Documentation/DistancePointEllipseEllipsoid.pdf

    we use the change of var, s = t/e1^2
    we also scale the target by axis, zi = yi/ei 
    let r0 = e0^2/e1^2 > 1
    requires e0 >= e1 > 0
    """
    z0 = y0/e0
    z1 = y1/e1
    r0 = (e0/e1)**2
    n0 = r0 * z0
    s0 = z1 - 1 # t0 = -e1^2+e1y1, s0 = -1 + y1/e1
    # inside the ellipse
    if (y0**2/e0**2 + y1**2/e1**2 - 1) < 0:
        s1 = 0
    else:
        s1 = np.linalg.norm([n0, z1]) - 1 # t1 = -e1^2 + sqrt(e0^2y0^2+e1^2y1^2), s1 = -1 + sqrt(e0^2y0^2/e1+e1y1^2)
    s = 0
    
    iteration = 0
    while iteration < maxit:
        s = (s0+s1)/2 # div in half
        if s == s0 or s == s1: # if average is the boundary, then we found the point
            break
        
        ratio0 = n0/(s+r0)
        ratio1 = z1/(s+1)
        g = ratio0**2 + ratio1**2 - 1
        if g > 0:
            s0 = s # if positive, replace lower bound
        elif g < 0:
            s1 = s # if negative, replace upper bound
        else:
            break 
            
        iteration += 1

    if iteration == 2000:
        print("Warning, maximum iteration reaeched, result may be unreliable.")
    
    return s * (e1**2)
            



if __name__ == "__main__":


    surface = lambda x: (2*x[0]/3)**2 + x[1]**2 + x[2]**2 - 1
    #testing
    print(min_point_ellpsoid((1,0,0), 3/2, 1), minimize.find_min((1,0,0), surface).x)
    print(min_point_ellpsoid((-1,-5,-3), 3/2, 1),  minimize.find_min((-1,-5,-3), surface).x) 
    print(min_point_ellpsoid((-1,3,9), 3/2, 1), minimize.find_min((-1,3,9), surface).x) 


    # start = time.time()
    # for _ in range(1000):
    #     min_point_ellpsoid((-1,-5,-3), 3/2, 1)
    # print(time.time()-start)
    

    print(min_point_ellpsoid((0.1,0.1,0.1), 3/2, 1), minimize.find_min((0.1,0.1,0.1), surface).x)
    print(min_point_ellpsoid((-0.1,0.1,0.1), 3/2, 1), minimize.find_min((-0.1,0.1,0.1), surface).x)
    print(min_point_ellpsoid((0.1,-0.1,0.1), 3/2, 1), minimize.find_min((0.1,-0.1,0.1), surface).x)
    print(min_point_ellpsoid((0.1,0.1,-0.1), 3/2, 1), minimize.find_min((0.1,0.1,-0.1), surface).x)
    print(min_point_ellpsoid((-0.1,-0.1,0.1), 3/2, 1), minimize.find_min((-0.1,-0.1,0.1), surface).x)
    print(min_point_ellpsoid((-0.1,0.1,-0.1), 3/2, 1), minimize.find_min((-0.1,0.1,-0.1), surface).x)
    print(min_point_ellpsoid((0.1,-0.1,-0.1), 3/2, 1), minimize.find_min((0.1,-0.1,-0.1), surface).x)
    print(min_point_ellpsoid((-0.1,-0.1,-0.1), 3/2, 1), minimize.find_min((-0.1,-0.1,-0.1), surface).x)


