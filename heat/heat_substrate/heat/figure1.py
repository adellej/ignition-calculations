%figure 1
import heat.heat3 as h
from heat.heat3 import Diffuse as D
d = D(nx=50, lx=.25j, ly=0.25j, y0=0.5j, ny=20, nz=20, r=1000, xhot=0.02j, thot=1.e8, geo='s', mbase=h.FLUX, mtop=h.BLACKBODY)
d.plot(nolabel=True, noaxes=True, nocb=True)
e = D(nx=50, lx=.25j, ly=0.25j, y0=0.5j, ny=20, nz=20, r=1000, xhot=0.02j, surf='x', thot=1.e8, geo='e', mbase=h.FLUX, mtop=h.BLACKBODY)
e.plot(nolabel=True, noaxes=True, nocb=True)