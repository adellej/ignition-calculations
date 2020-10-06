import numpy as np

from numpy import cbrt

def qqrt(x):
    return np.sqrt(np.sqrt(x))


def quad(x):
    return (x**2)**2

def cube(x):
    return (x**2)*x

TINY = np.finfo('float64').tiny
