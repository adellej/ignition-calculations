import fipy
import numpy as np
import matplotlib.pyplot as plt

# fipy example from https://math.nist.gov/mcsd/Seminars/2005/2005-03-01-wheeler-presentation.pdf

nx = 10
dx = 1
steps = 100

# create a mesh:
L = nx * dx
from fipy.meshes.grid2D import Grid2D

mesh = Grid2D(nx=nx, dx=dx)

# create a field variable, set the initial conditions:
from fipy.variables.cellVariable import CellVariable

var = CellVariable(mesh=mesh, value=0)


def centerCells(cell):
    return abs(cell.getCenter()[0] - L / 2.0) < L / 10


var.setValue(value=1.0, cells=mesh.getCells(filter=centerCells))

# create the equation:
from fipy.terms.transientTerm import TransientTerm
from fipy.terms.implicitDiffusionTerm import ImplicitDiffusionTerm

eq = TransientTerm() - ImplicitDiffusionTerm(coeff=1) == 0

# create a viewer:
from fipy.viewers.gist2DViewer import Gist1DViewer

viewer = Gist1DViewer(vars=(var,), limits=('e', 'e', 0, 1))
viwer.plot()

# solve
for i in range(steps):
    var.updateOld()
    eq.solve()
    viewer.plot()
