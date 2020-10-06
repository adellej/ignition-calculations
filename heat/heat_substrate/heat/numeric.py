import numpy as np

from numpy import cbrt
from numpy import sqrt

def qqrt(x):
    return sqrt(sqrt(x))

def quad(x):
    return (x**2)**2

def cube(x):
    return (x**2)*x

TINY = np.finfo('float64').tiny

GOLDEN = 0.5 * (sqrt(5.) + 1.)
