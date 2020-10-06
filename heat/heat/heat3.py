import time  # always good to have
import resource
import copy
import pickle
import gzip
import lzma
import bz2
from pathlib import Path

from functools import partial
from uuid import uuid1

import numpy as np

from scipy.optimize import root
from scipy.linalg import solve
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse import diags
from matplotlib import pylab as plt
from matplotlib.colors import LogNorm, Normalize
from numpy import imag
from numpy import real
from numpy.linalg import norm

from human import time2human
import color

from physconst import KB, RK
from movie import MovieWriter
from movie import make_movie
from movie import MPLCanvas

from kepdump import load as D

from isotope import ion as I

from net import Net3aFFNs, Net3aFFNt, Net3aC12, NetNone
from kappa import kappa

from numeric import *

REFLECTIVE = 0
PERIODIC = 1
TEMPERATURE = 2
FLUX = 3
CONNECTXY = 4
BLACKBODY = 5
CUSTOM = -1


def vec(a):
    shape = tuple(np.max([x.shape for x in a], axis=0)) + (len(a),)
    v = np.empty(shape)
    for i, x in enumerate(a):
        v[..., i] = x
    return v


class _Diffuse_Store(object):
    pass


class _Diffuse_Result(object):
    pass


class Diffuse(object):
    def __init__(
        self,
        nx=80,
        ny=1,
        nz=30,
        lx=400,
        ly=1,
        lz=150,
        x0=0,
        y0=0,
        z0=0,
        r=1e6,
        geo='cart',
        mbase=FLUX,
        mtop=BLACKBODY,  # BLACKBODY | TEMPERATURE
        surf='r',
        tbase=1e8,
        fbase=1e19,
        ttop=1e6,
        thot=5e6,
        xhot=20,
        tol=1e-6,
        op=None,
        rhoflag='fit',
        method='solve4',
        normalize=False,
        T=None,
        sparse=True,
        initial=False,
        jac=True,
        xres='constant',  # constant | progressive
        yres='constant',  # ...
        zres='constant',  # ... | logarithmic | surflog
        root_options=None,
        root4=True,
        Y=1,
        net='static',
        bc=None,
        logfac=0.1,
        kappa='~/python/heat/kappa.npz',  # <filename> | value | None
        dump=None,
        run_name = 'test',
    ):

        if initial is None:
            return

        if dump is not None:
            if isinstance(dump, (str, Path)):
                dump = D(dump)
            # as first test, let's just take grid, add interpolation later
            j0 = np.maximum(dump.jshell0, dump.jburnmin)
            j1 = dump.jm
            sel = slice(j0, j1 + 1)
            dz1 = dump.dr[sel]
            nz = len(dz1)
            zres = 'dump'
            zf1 = np.ndarray(nz + 1)
            zf1[0] = z0
            zf1[1:] = z0 + np.cumsum(dz1)
            zc1 = 0.5 * (zf1[1:] + zf1[:-1])
            self.r = dump.rn[j0 - 1]
            tbase = 0.5 * (dump.tn[j0] + dump.tn[j0 - 1])
            if j0 == 1:
                fbase = dump.xlum0
            else:
                fbase = dump.xln[j0 - 1]
            fbase /= 4 * np.pi * dump.rn[j0 - 1] ** 2
            ttop = dump.teff
            rhoflag = 'dump'
            rhoc1 = dump.dn[sel]
            rhof1 = np.ndarray((nz + 1,))
            rhof1[1:-1] = 0.5 * (rhoc1[1:] + rhoc1[:-1])
            if j0 > 0:
                rhof1[0] = 0.5 * (rhoc1[j0] + rhoc1[j0 - 1])
            else:
                rhof1[0] = rhoc[0]
            if j1 >= dump.jm:
                rhof1[-1] = rhoc1[-1]
            else:
                rhof1[-1] = 0.5 * (rhoc1[j1] + rhoc1[j1 + 1])
            tc1 = dump.tn[sel]
        self.dump = dump

        # we use 'f' for face and 'c' for centre

        nx1 = nx + 1
        ny1 = ny + 1
        nz1 = nz + 1

        self.shape = (nx, ny, nz)

        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.nx1 = nx1
        self.ny1 = ny1
        self.nz1 = nz1

        self.run_name = run_name

        if ly == 'lx':
            ly = lx

        self.lx = lx
        self.ly = ly
        self.lz = lz

        self.bc = np.ndarray((3, 2), dtype=np.object)
        # boundary conditions
        # self.bc = [PERIODIC, PERIODIC, CUSTOM]
        # self.bc = [PERIODIC, REFLECTIVE, CUSTOM]
        # self.bc = [REFLECTIVE, PERIODIC, CUSTOM]
        if bc is None:
            self.bc[...] = np.array([REFLECTIVE, REFLECTIVE, CUSTOM]).reshape(-1, 1)
            self.bc[2, 1] = mtop
            self.bc[2, 0] = mbase

        # set up network
        self.Y = Y
        if net == 'static':
            net = Net3aFFNs
            abu = (Y / 4,)
        elif net == 'dynamic':
            net = Net3aFFNt
            abu = (Y / 4, (1 - Y) / 12)
        elif net == 'fancy':
            net = Net3aC12
            abu = (Y / 4, (1 - Y) / 12, 0)
        elif net is None:
            net = NetNone
            abu = None
        else:
            raise AttributeError(f'Unknown network type: {net}')
        self.net = net(abu, self.shape)

        # basic 1D linear coordinate grid through (0,0), naming: ends in "1"

        if xres == 'progressive':
            nnx = (nx * (nx + 1)) // 2
            dx1 = lx * (np.arange(nx) + 1) / nnx
        elif xres == 'constant':
            dx1 = np.tile(lx / nx, nx)
        else:
            raise AttributeError(f'Unknown xres = {xres}')
        xf1 = x0 + np.insert(np.cumsum(dx1), 0, 0)
        xc1 = 0.5 * (xf1[1:] + xf1[:-1])

        if yres == 'progressive':
            nny = (ny * (ny + 1)) // 2
            dy1 = ly * (np.arange(ny) + 1) / nny
        elif yres == 'constant':
            dy1 = np.tile(ly / ny, ny)
        else:
            raise AttributeError(f'Unknown yres = {yres}')
        yf1 = y0 + np.insert(np.cumsum(dy1), 0, 0)
        yc1 = 0.5 * (yf1[1:] + yf1[:-1])

        if zres == 'progressive':
            nnz = (nz * (nz + 1)) // 2
            dz1 = lz * (np.arange(nz)[::-1] + 1) / nnz
        elif zres == 'logarithmic':
            r0 = r + z0
            r1 = r + z0 + lz
            f = (r1 / r0) ** (1 / nz)
            zf1 = r0 * f ** np.arange(nz + 1) - r0
            dz1 = r0 * (f - 1) * f ** np.arange(nz)
            zc1 = r0 * np.sqrt(f) * f ** np.arange(nz)
        elif zres == 'surflog':
            logfac = 1 + logfac
            size = (logfac ** nz - 1) / (logfac - 1)
            dz1 = logfac ** np.arange(nz)[::-1] * lz / size
        elif zres == 'constant':
            dz1 = np.tile(lz / nz, nz)
        elif zres == 'dump':
            pass
        else:
            raise AttributeError(f'Unknown zres = {zres}')
        if zres in ('progressive', 'constant', 'surflog',):
            zf1 = z0 + np.insert(np.cumsum(dz1), 0, 0)
            zc1 = 0.5 * (zf1[1:] + zf1[:-1])

        self.sparse = sparse
        self.jac = jac

        self.r = r
        self.fbase = fbase
        self.tbase = tbase
        self.mbase = mbase
        self.ttop = ttop
        self.thot = thot
        self.xhot = xhot
        self.surf = surf

        self.rhoflag = rhoflag
        self.kb = 1.231e15 * (2 ** (4 / 3))  # ideal gas constant for pure helium
        self.g = 2.426e14  # surface gravity, cgs
        self.ARAD = 7.566e-15  # radiation constant
        self.CLIGHT = 2.998e10  # speed of light
        self.SB = 5.6704e-5

        if kappa == None:
            self._kappa = 0.136  # cm^2/g, assume constant opacity
        else:
            self._kappa = kappa
        self.const_kappa = not isinstance(self._kappa, str)

        self.cst4 = -self.ARAD * self.CLIGHT / 3
        self.cst = 4 * self.cst4

        if dump is None:
            rhof1 = self.rho_ini(zf1)
            rhoc1 = self.rho_ini(zc1)

        rhofi1 = 1 / rhof1
        rhoci1 = 1 / rhoc1

        rhoc = np.tile(rhoc1, (nx, ny, 1))
        rhoci = np.tile(rhoci1, (nx, ny, 1))

        rhoz = np.tile(rhof1, (nx, ny, 1))
        rhozi = np.tile(rhofi1, (nx, ny, 1))

        rhox = np.tile(rhoc1, (nx1, ny, 1))
        rhoxi = np.tile(rhoci1, (nx1, ny, 1))

        rhoy = np.tile(rhoc1, (nx, ny1, 1))
        rhoyi = np.tile(rhoci1, (nx, ny1, 1))

        # geometric factors:
        #   az, ay, az - areas relative on sides
        #   v - volumes
        #   .c - centre coordinates
        #   .f - corner coordinates of faces
        #   .d - face centre coordinates in .-direction
        dz = np.tile(dz1.reshape(1, 1, -1), (nx1, ny1, 1))
        zf = np.tile(zf1.reshape(1, 1, -1), (nx1, ny1, 1))
        zd = np.tile(zf1.reshape(1, 1, -1), (nx, ny, 1))
        zc = np.tile(zc1.reshape(1, 1, -1), (nx, ny, 1))

        ax = np.ndarray((nx1, ny, nz))
        ay = np.ndarray((nx, ny1, nz))
        az = np.ndarray((nx, ny, nz1))
        v = np.ndarray((nx, ny, nz))
        if geo in ('d', 'cart', 'cartesian',):
            # 'd' for 'Decartes'

            dx = np.tile(dx1.reshape(-1, 1, 1), (1, ny1, nz1))
            xf = np.tile(xf1.reshape(-1, 1, 1), (1, ny1, nz1))
            xd = np.tile(xf1.reshape(-1, 1, 1), (1, ny, nz))
            xc = np.tile(xc1.reshape(-1, 1, 1), (1, ny, nz))

            dy = np.tile(dy1.reshape(1, -1, 1), (nx1, 1, nz1))
            yf = np.tile(yf1.reshape(1, -1, 1), (nx1, 1, nz1))
            yd = np.tile(yf1.reshape(1, -1, 1), (nx, 1, nz))
            yc = np.tile(yc1.reshape(1, -1, 1), (nx, 1, nz))

            # plotting coordinates 3D vectors
            vz0 = np.array([0, 0, zf1[0]])

            def c2v(x, y, z):
                x = x.reshape(-1, 1, 1)
                y = y.reshape(1, -1, 1)
                z = z.reshape(1, 1, -1)
                return vec((x, y, z))

            self.vz0 = vz0
            self.vf = c2v(xf1, yf1, zf1) - vz0
            self.vc = c2v(xc1, yc1, zc1) - vz0
            self.vxe = c2v(xc1, yf1, zf1) - vz0
            self.vye = c2v(xf1, yc1, zf1) - vz0
            self.vze = c2v(xf1, yf1, zc1) - vz0
            self.vxb = c2v(xf1, yc1, zc1) - vz0
            self.vyb = c2v(xc1, yf1, zc1) - vz0
            self.vzb = c2v(xc1, yc1, zf1) - vz0

            ax[...] = dy[:, :, :-1] * dz[:, :-1, :]
            ay[...] = dz[:-1, :, :] * dx[:, :, :-1]
            az[...] = dx[:, :-1, :] * dy[:-1, :, :]
            v[...] = dx[:, :-1, :-1] * dy[:-1, :, :-1] * dz[:-1, :-1, :]
            geo = 'cartesian'
        elif geo in ('c', 'cr', 'cyl', 'cylindrical',):
            p2 = 2 * np.pi

            if imag(ly) != 0:
                assert real(ly) == 0
                ly = imag(ly)
                dy1 = imag(dy1)
                yf1 = imag(yf1)
                yc1 = imag(yc1)
            else:
                f = 1 / (p2 * xf1[-1])
                dy1 *= f
                yf1 *= f
                yc1 *= f
                ly *= f
            assert 0 < ly <= 1

            dx = np.tile(dx1.reshape(-1, 1, 1), (1, ny1, nz1))
            xf = np.tile(xf1.reshape(-1, 1, 1), (1, ny1, nz1))
            xd = np.tile(xf1.reshape(-1, 1, 1), (1, ny, nz))
            xc = np.tile(xc1.reshape(-1, 1, 1), (1, ny, nz))

            # we interpret y in units of 2 * pi and convert to physical arch length
            dy = np.tile(dy1.reshape(1, -1, 1), (nx1, 1, nz1))
            yf = np.tile(yf1.reshape(1, -1, 1), (nx1, 1, nz1))
            yd = np.tile(yf1.reshape(1, -1, 1), (nx, 1, nz))
            yc = np.tile(yc1.reshape(1, -1, 1), (nx, 1, nz))

            dy *= xf[:, :1, :1] * p2
            yf *= xf[:, :1, :1] * p2
            yd *= xc[:, :1, :1] * p2
            yc *= xc[:, :1, :1] * p2

            # set to maximum disk radius
            self.ly = yf[-1, -1, -1] - yf[-1, 0, -1]

            # plotting coordinates 3D vectors
            def c2v(x, y, z):
                # x is proejected surface arc from pole, y is phase
                phi = y.reshape(1, -1, 1) * p2
                cp = np.cos(phi)
                sp = np.sin(phi)
                x = x.reshape(-1, 1, 1)
                z = z.reshape(1, 1, -1)
                return vec((x * sp, x * cp, z,))

            vz0 = np.array([0, 0, zf1[0]])
            self.vz0 = vz0
            self.vf = c2v(xf1, yf1, zf1) - vz0
            self.vc = c2v(xc1, yc1, zc1) - vz0
            self.vxe = c2v(xc1, yf1, zf1) - vz0
            self.vye = c2v(xf1, yc1, zf1) - vz0
            self.vze = c2v(xf1, yf1, zc1) - vz0
            self.vxb = c2v(xf1, yc1, zc1) - vz0
            self.vyb = c2v(xc1, yf1, zc1) - vz0
            self.vzb = c2v(xc1, yc1, zf1) - vz0

            fz = 0.5 * xf[:, :-1, :] * dy[:, :, :]
            dfz = fz[1:] - fz[:-1]
            az[...] = dfz
            ay[...] = dz[:-1, :, :] * dx[:, :, :-1]
            ax[...] = dy[:, :, :-1] * dz[:, :-1, :]
            v[...] = az[:, :, :-1] * dz[:-1, :-1, :]

            self.bc[0][:] = REFLECTIVE
            if geo == 'cr':
                self.bc[1][:] = REFLECTIVE
            else:
                self.bc[1][:] = PERIODIC
            geo = 'cylindrical'
        elif geo in ('s', 'sr', 'sph', 'spherical', 'e', 'er', 'equ', 'equatorial',):
            # use coordinates relative to surface (to allow full sphere)
            r1 = r + zf1[-1]
            r1i = 1 / r1
            p2 = 2 * np.pi
            if imag(lx) != 0:
                assert real(lx) == 0
                lx = imag(lx) * r1 * p2
                dx1 = imag(dx1) * r1 * p2
                xf1 = imag(xf1) * r1 * p2
                xc1 = imag(xc1) * r1 * p2
            assert lx > 0

            if imag(ly) != 0:
                assert real(ly) == 0
                ly = imag(ly)
                dy1 = imag(dy1)
                yf1 = imag(yf1)
                yc1 = imag(yc1)
            else:
                phi1 = xf1 * r1i
                assert phi1[0] >= 0
                assert phi1[-1] <= np.pi
                assert phi1[0] < phi1[1]
                if phi1[-1] <= np.pi * 0.5:
                    f1 = xf1[-1]
                elif phi1[0] >= np.pi * 0.5:
                    f1 = xf1[0]
                else:
                    f1 = np.pi * 0.5 * r1
                f = 1 / (p2 * f1)
                ly *= f
                dy1 *= f
                yf1 *= f
                yc1 *= f
            assert 0 < ly <= 1

            if geo in ('e', 'er', 'equ', 'equatorial'):
                geo_ = 'equatorial'
                assert x0 >= -0.5 * r1 * np.pi
                assert x0 + lx <= 0.5 * r1 * np.pi
                p0 = 0.5
            else:
                geo_ = 'spherical'
                assert x0 >= 0
                assert x0 + lx <= r1 * np.pi
                p0 = 0
            p1 = np.pi
            ph = 0.5 * np.pi
            a0 = p0 * p2

            rf1 = ((zf1 + r) * r1i).reshape(1, 1, -1)
            rc1 = ((zc1 + r) * r1i).reshape(1, 1, -1)
            dx = np.tile(dx1.reshape(-1, 1, 1) * rf1, (1, ny1, 1))
            xf = np.tile(xf1.reshape(-1, 1, 1) * rf1, (1, ny1, 1))
            xd = np.tile(xf1.reshape(-1, 1, 1) * rc1, (1, ny, 1))
            xc = np.tile(xc1.reshape(-1, 1, 1) * rc1, (1, ny, 1))

            # we interpret y in units of 2 * pi and convert to physical arc length
            sf1 = np.sin(xf1.reshape(-1, 1, 1) * r1i + a0) * r1
            sc1 = np.sin(xc1.reshape(-1, 1, 1) * r1i + a0) * r1
            dy = dy1.reshape(1, -1, 1) * p2 * rf1 * sf1
            yf = yf1.reshape(1, -1, 1) * p2 * rf1 * sf1
            yd = yf1.reshape(1, -1, 1) * p2 * rc1 * sc1
            yc = yc1.reshape(1, -1, 1) * p2 * rc1 * sc1

            # plotting coordinates 3D vectors
            vz0 = np.array([0, 0, self.r + zf1[0]])

            def c2v(x, y, z):
                # x is projected surface arc from pole, y is phase
                theta = x.reshape(-1, 1, 1) * r1i
                phi = y.reshape(1, -1, 1) * p2
                ct = np.cos(theta)
                st = np.sin(theta)
                cp = np.cos(phi)
                sp = np.sin(phi)
                z = z + r
                return vec((z * st * sp, z * st * cp, z * ct,))

            self.vz0 = vz0
            self.vf = c2v(xf1, yf1, zf1) - vz0
            self.vc = c2v(xc1, yc1, zc1) - vz0
            self.vxe = c2v(xc1, yf1, zf1) - vz0
            self.vye = c2v(xf1, yc1, zf1) - vz0
            self.vze = c2v(xf1, yf1, zc1) - vz0
            self.vxb = c2v(xf1, yc1, zc1) - vz0
            self.vyb = c2v(xc1, yf1, zc1) - vz0
            self.vzb = c2v(xc1, yc1, zf1) - vz0

            if p0 < 0.25:
                h = (1 - np.cos(xf1 * r1i + a0)).reshape(-1, 1, 1) * rf1
            else:
                h = np.sin(xf1 * r1i + (ph - a0)).reshape(-1, 1, 1) * rf1
            fz = h * (r + zf1).reshape(1, 1, -1) * p2 * dy1.reshape(1, -1, 1)
            az[...] = fz[1:, :, :] - fz[:-1, :, :]

            w = np.sin(xf1 * r1i + a0).reshape(-1, 1, 1) * rf1
            fx = w * (r + zf1).reshape(1, 1, -1) * p1 * dy1.reshape(1, -1, 1)
            ax[...] = fx[:, :, 1:] - fx[:, :, :-1]

            fy = 0.5 * dx * zf1.reshape(1, 1, -1)
            ay[...] = fy[:, :, 1:] - fy[:, :, :-1]

            vt = az[:, :, :] * rf1 * (1 / 3)
            v[:, :] = vt[:, :, 1:] - vt[:, :, :-1]

            self.bc[0][:] = REFLECTIVE
            if geo in ('sr', 'er'):
                self.bc[1][:] = REFLECTIVE
            else:
                self.bc[1][:] = PERIODIC
            geo = geo_
        elif geo in ('p', 'patch', 'pc', 'pe',):
            # assume polar coordinates relative to (0,0,1) as rectangular achor point, with poles at (0,1,0) and (1,0,0)
            r1 = r + zf1[-1]
            r1i = 1 / r1
            p2 = 2 * np.pi
            p2i = 1 / p2
            cfi = r1i * p2i

            if imag(lx) != 0:
                assert real(lx) == 0
                lx = imag(lx)
                dx1 = imag(dx1)
                xf1 = imag(xf1)
                xc1 = imag(xc1)
            else:
                f = cfi
                lx *= f
                dx1 *= f
                xf1 *= f
                xc1 *= f
            assert 0 < lx <= 1

            if imag(ly) != 0:
                assert real(ly) == 0
                ly = imag(ly)
                dy1 = imag(dy1)
                yf1 = imag(yf1)
                yc1 = imag(yc1)
            else:
                f = cfi
                ly *= f
                dy1 *= f
                yf1 *= f
                yc1 *= f
            assert 0 < ly <= 1

            if geo not in ('pe',):
                xf1 -= lx * 0.5
                yf1 -= ly * 0.5
                xc1 -= lx * 0.5
                yc1 -= ly * 0.5

            rf1 = (self.r + zf1).reshape(1, 1, -1)
            rc1 = (self.r + zc1).reshape(1, 1, -1)

            def spherical_excess(a, b, c):
                return 4 * np.arctan(
                    np.sqrt(
                        np.maximum(
                            0,
                            np.tan(0.25 * (a + b + c))
                            * np.tan(0.25 * (-a + b + c))
                            * np.tan(0.25 * (-b + a + c))
                            * np.tan(0.25 * (-c + a + b)),
                        )
                    )
                )

            def arc(x, y, r, geodesic=True, surf=False, coord=False):
                dot = partial(np.einsum, "...i,...i->...")
                sx = np.sin(x * p2).reshape(-1, 1, 1)
                cx = np.cos(x * p2).reshape(-1, 1, 1)
                sy = np.sin(y * p2).reshape(1, -1, 1)
                cy = np.cos(y * p2).reshape(1, -1, 1)

                z3 = np.zeros((1, 1, 1))

                vny = vec([cx, z3, -sx])
                vnx = vec([z3, cy, -sy])

                vax = vec([z3, sy, cy])
                vay = vec([sx, z3, cx])

                vp = vec([cy * sx, cx * sy, cx * cy])
                vn = vp / (norm(vp, axis=-1)[..., np.newaxis] + TINY)
                result = tuple()
                if geodesic or surf:
                    bx = np.arctan2(dot(np.cross(vax, vn), vnx), dot(vax, vn))
                    by = np.arctan2(dot(np.cross(vn, vay), vny), dot(vn, vay))
                if geodesic:
                    result += (
                        bx * r,
                        by * r,
                    )
                if surf:
                    a = bx[1:, :, :] - bx[:-1, :, :]
                    b = by[:, 1:, :] - by[:, :-1, :]
                    c = np.arcsin(
                        norm(np.cross(vn[1:, :-1, :], vn[:-1, 1:, :]), axis=-1)
                    )
                    a1 = a[:, :-1, :]
                    a2 = a[:, 1:, :]
                    b1 = b[:-1, :, :]
                    b2 = b[1:, :, :]
                    omega = spherical_excess(a1, b1, c) + spherical_excess(a2, b2, c)
                    result += (omega * r ** 2,)
                if coord:
                    result += (vn * r[..., np.newaxis],)
                if len(result) == 1:
                    return result[0]
                return result

            xf, yf, az, self.vf = arc(xf1, yf1, rf1, surf=True, coord=True)
            dx = xf[1:, :, :] - xf[:-1, :, :]
            dy = yf[:, 1:, :] - yf[:, :-1, :]
            xc, yc, self.vc = arc(xc1, yc1, rc1, coord=True)
            xd, _ = arc(xf1, yc1, rc1)
            _, yd = arc(xc1, yf1, rc1)

            # for plotting (real cartesian coordinates)
            vz0 = np.array([0, 0, self.r + z0])
            self.vxe = arc(xc1, yf1, rf1, geodesic=False, coord=True) - vz0
            self.vye = arc(xf1, yc1, rf1, geodesic=False, coord=True) - vz0
            self.vze = arc(xf1, yf1, rc1, geodesic=False, coord=True) - vz0

            self.vxb = arc(xf1, yc1, rc1, geodesic=False, coord=True) - vz0
            self.vyb = arc(xc1, yf1, rc1, geodesic=False, coord=True) - vz0
            self.vzb = arc(xc1, yc1, rf1, geodesic=False, coord=True) - vz0
            self.vz0 = vz0
            self.vf -= vz0
            self.vc -= vz0

            fx = 0.5 * dy * rf1.reshape(1, 1, -1)
            ax[...] = fx[:, :, 1:] - fx[:, :, :-1]
            fy = 0.5 * dx * rf1.reshape(1, 1, -1)
            ay[...] = fy[:, :, 1:] - fy[:, :, :-1]

            vz = az[:, :, :] * rf1 * (1 / 3)
            v[:, :] = vz[:, :, 1:] - vz[:, :, :-1]

            self.v = v

            geo = 'patch'
            self.bc[0:1][:] = REFLECTIVE
        else:
            raise AttributeError(f'Unsupported Geometry "{geo}".')

        # lower-dimensional cases:  (for now)
        if nx == 1:
            self.bc[0][:] = REFLECTIVE
        if ny == 1:
            self.bc[1][:] = REFLECTIVE

        # why not use d(xc, yc, zc)?
        dxf = np.ndarray((nx1, ny, nz))
        dxf[1:-1, :, :] = 0.125 * (
            dx[1:, 1:, 1:]
            + dx[:-1, 1:, 1:]
            + dx[1:, :-1, 1:]
            + dx[:-1, :-1, 1:]
            + dx[1:, 1:, :-1]
            + dx[:-1, 1:, :-1]
            + dx[1:, :-1, :-1]
            + dx[:-1, :-1, :-1]
        )
        dxf[0, :, :] = 0.25 * (
            dx[0, :-1, :-1] + dx[0, 1:, :-1] + dx[0, :-1, 1:] + dx[0, 1:, 1:]
        )
        dxf[-1, :, :] = 0.25 * (
            dx[-1, :-1, :-1] + dx[-1, 1:, :-1] + dx[-1, :-1, 1:] + dx[-1, 1:, 1:]
        )
        self.dxfi = 1 / (dxf + TINY)

        dyf = np.ndarray((nx, ny1, nz))
        dyf[:, 1:-1, :] = 0.125 * (
            dy[1:, 1:, 1:]
            + dy[:-1, 1:, 1:]
            + dy[1:, :-1, 1:]
            + dy[:-1, :-1, 1:]
            + dy[1:, 1:, :-1]
            + dy[:-1, 1:, :-1]
            + dy[1:, :-1, :-1]
            + dy[:-1, :-1, :-1]
        )
        dyf[:, 0, :] = 0.25 * (
            dy[:-1, 0, :-1] + dy[1:, 0, :-1] + dy[:-1, 0, 1:] + dy[1:, 0, 1:]
        )
        dyf[:, -1, :] = 0.25 * (
            dy[:-1, -1, :-1] + dy[1:, -1, :-1] + dy[:-1, -1, 1:] + dy[1:, -1, 1:]
        )
        self.dyfi = 1 / (dyf + TINY)

        dzf = np.ndarray((nx, ny, nz1))
        dzf[:, :, 1:-1] = 0.125 * (
            dz[1:, 1:, 1:]
            + dz[:-1, 1:, 1:]
            + dz[1:, :-1, 1:]
            + dz[:-1, :-1, 1:]
            + dz[1:, 1:, :-1]
            + dz[:-1, 1:, :-1]
            + dz[1:, :-1, :-1]
            + dz[:-1, :-1, :-1]
        )
        dzf[:, :, 0] = 0.25 * (
            dz[:-1, :-1, 0] + dz[1:, :-1, 0] + dz[:-1, 1:, 0] + dz[1:, 1:, 0]
        )
        dzf[:, :, -1] = 0.25 * (
            dz[:-1, :-1, -1] + dz[1:, :-1, -1] + dz[:-1, 1:, -1] + dz[1:, 1:, -1]
        )
        self.dzfi = 1 / (dzf + TINY)

        m = v * rhoc

        v0 = v[0, 0, 0]
        f0 = v0 ** (2 / 3)
        if normalize:
            #   f0 - area to which units are normalised
            #   v0 - v[0,0]
            f0i = 1 / f0
            # v0i = 1 / v0
            ax *= f0i
            ay *= f0i
            az *= f0i
            # v *= f0i
            m *= f0i

        self.geo = geo
        self.ax = ax
        self.ay = ay
        self.az = az
        # self.v = v
        # self.f0 = f0
        # self.v0 = v0
        # self.vl = v0 / f0
        self.m = m
        self.rhoc = rhoc
        self.rhox = rhox
        self.rhoy = rhoy
        self.rhoz = rhoz
        self.rhoxi = rhoxi
        self.rhoyi = rhoyi
        self.rhozi = rhozi

        self.xf = xf
        self.yf = yf
        self.zf = zf

        self.xc = xc
        self.yc = yc
        self.zc = zc

        self.xd = xd
        self.yd = yd
        self.zd = zd

        self.xcsurf = 0.25 * (
            xf[1:, 1:, -1] + xf[:-1, 1:, -1] + xf[1:, :-1, -1] + xf[:-1, :-1, -1]
        )
        self.ycsurf = 0.25 * (
            yf[1:, 1:, -1] + yf[:-1, 1:, -1] + yf[1:, :-1, -1] + yf[:-1, :-1, -1]
        )
        # r will depend on geometry and reference location
        if self.geo in ('cartesian', 'patch',):
            self.rcsurf = np.sqrt(self.xcsurf ** 2 + self.ycsurf ** 2)
        elif self.geo in ('cylindrical', 'spherical'):
            # more tricky if p0 != 0
            self.rcsurf = self.xcsurf

        self.Cv = m * 1.5 * RK * (3 / 4)

        # initialise T
        if dump is not None:
            T = np.tile(tc1.reshape(1, 1, -1), (nx, ny, 1))
        elif T is None or T.shape != v.shape:
            # initialise T (crudely)
            dcd = rhoz * dzf
            cd = np.cumsum(dcd[:, :, ::-1], axis=-1)[:, :, ::-1]
            cd = np.dstack((cd, np.zeros((nx, ny, 1))))
            cdc = cd[:, :, 1:-1]

            ttop = self.ttop_func(T4=True)
            if mbase == FLUX:
                try:
                    kappa = float(self._kappa)
                except:
                    kappa = 0.2
                T4 = (
                    3 * kappa * fbase * cdc / (self.ARAD * self.CLIGHT)
                    + ttop[:, :, np.newaxis]
                )
            else:
                T4 = (
                    quad(tbase) * cdc / cd[:, :, 0][:, np.newaxis]
                    + ttop[:, :, np.newaxis]
                )
            T = qqrt(T4)

        self.method = method

        if initial:
            self.T = T
            return

        # solve
        starttime = time.time()
        if method in (
            'df-sane',
            'lm',
            'broyden1',
            'broyden2',
            'diagbroyden',
            'linearmixing',
            'excitingmixing',
            'anderson',
            'hybr',
        ):
            if method in (
                'df-sane',
                'broyden1',
                'broyden2',
                'diagbroyden',
                'linearmixing',
                'excitingmixing',
                'anderson',
            ):
                self.jac = None
            if method in ('lm', 'hybr',):
                if self.jac is True:
                    self.sparse = False

            t = np.ndarray((2, 3))
            use = resource.getrusage(resource.RUSAGE_SELF)
            t[0, :] = (time.time(), use.ru_utime, use.ru_stime)
            T_ = T.flatten()
            if root4:
                T_ = quad(T_)
                fun = partial(self.func, T4=True)
            else:
                fun = partial(self.func, T4=False)
            solution = root(fun, T_, method=method, jac=jac, options=dict())
            if root4:
                solution.x = qqrt(solution.x)
            use = resource.getrusage(resource.RUSAGE_SELF)
            t[1, :] = (time.time(), use.ru_utime, use.ru_stime)
            dt = t[1] - t[0]
            x = np.hstack((dt, np.sum(dt[1:])))
            print(
                f'[DIFFUSE] [{method}] total:   time: {time2human(x[0]):>8s} real, {time2human(x[1]):>8s} user, {time2human(x[2]):>8s} sys, {time2human(x[3]):>8s} cpu total.'
            )
        elif method == 'solve4':
            solution = self.solve(quad(T), f=partial(self.func, T4=True), op=op)
            solution.x = np.sqrt(np.sqrt(solution.x))
        elif method == 'solve':
            method == 'solve'
            solution = self.solve(T, f=partial(self.func, T4=False), op=op)
        else:
            raise AttributeError('Unknown solver.')

        self.method = method
        self.solution = solution

        print(f'[DIFFUSE] message:    {solution.message}')
        print(f'[DIFFUSE] success:    {solution.success}')
        try:
            print(f'[DIFFUSE] iterations: {solution.nfev}')
        except:
            pass
        try:
            print(f'[DIFFUSE] iterations: {solution.nit}')
        except:
            pass

        runtime = time.time() - starttime
        print(f'[DIFFUSE] time:       {time2human(runtime)}')

        self.T = solution.x.reshape(self.shape)

        self.uuid = uuid1()

    def kappai_func(self, T, rho, T4=True):
        if self.const_kappa:
            if not hasattr(self, 'kappai'):
                self.kappai = (np.array(1 / self._kappa), np.array(0.0))
            return self.kappai

        if not hasattr(self, 'kappa_data'):
            self.kappa_data = kappa(self._kappa)
        kappai, kappait = self.kappa_data(T, rho, T4)

        # Test
        # kappai = np.log(T) * 0.01
        # kappait = 0.5 * 0.01 / (T * kappai)

        # kappai[:] = 5
        # kappait[:] = 0

        return kappai, kappait

    def ihot_func(self, reset=False):
        if reset:
            del self._ihot
        if hasattr(self, '_ihot'):
            return self._ihot
        # change depending on geometry
        if self.surf == 'x':
            ihot = np.where(self.xcsurf < self.xhot)
        elif self.surf == 'r':
            ihot = np.where(self.rcsurf < self.xhot)
        elif self.surf == 'y':
            ihot = np.where(self.ycsurf < self.xhot)
        elif self.surf == 'spots':
            if self.geo == 'cartesian':
                a = self.lx * self.ly
            elif self.geo == 'cylindrical':
                a = 0.5 * self.lx * self.ly
            f = a / self.xhot ** 2 * (0.5 * (np.sqrt(5) - 1))
            n = 1 + np.random.randint(1 + np.sqrt(f))
            print(f'[IHOT_FUNC] creating {n} spots.')
            xc = np.vstack((-self.xcsurf[::-1], self.xcsurf, self.xcsurf + self.lx))
            xc = np.hstack([xc] * 3)
            yc = np.hstack((-self.ycsurf[:, ::-1], self.ycsurf, self.ycsurf + self.ly))
            if self.geo == 'cylindrical':
                yc[:, 2 * self.ny :] = (self.ly / self.lx) * self.xc[
                    :, -1:, -1
                ] + self.ycsurf
            yc = np.vstack([yc] * 3)
            if self.geo == 'cylindrical':
                p = yc / xc
                d = xc
                xc = d * np.cos(p)
                yc = d * np.sin(p)
            ihot = np.tile(False, (3 * self.nx, 3 * self.ny))
            for i in range(n):
                if self.geo == 'cartesian':
                    x, y, r = np.random.random(3) * np.array([self.lx, self.ly, 1])
                elif self.geo == 'cylindrical':
                    d, p, r = np.random.random(3) * np.array(
                        [self.lx, self.ly / self.lx, 1]
                    )
                    x = d * np.cos(p)
                    y = d * np.sin(p)
                r = 10 ** -r * self.xhot
                ihot |= ((xc - x) ** 2 + (yc - y) ** 2) < (r ** 2)
                print(f'[SPOTS] Spot {i}: x = {x}, y = {y}, r = {r}')
            if self.bc[0] == REFLECTIVE:
                ihot = ihot[self.nx : 2 * self.nx, :]
            else:
                ihot = (
                    ihot[: self.nx, :]
                    | ihot[self.nx : 2 * self.nx, :]
                    | ihot[2 * self.nx :, :]
                )
            if self.bc[1] == REFLECTIVE:
                ihot = ihot[:, self.ny : 2 * self.ny]
            else:
                ihot = (
                    ihot[:, : self.ny]
                    | ihot[:, self.ny : 2 * self.ny]
                    | ihot[:, 2 * self.ny :]
                )
        self._ihot = ihot
        return ihot

    def ttop_func(self, T4):
        ihot = self.ihot_func()
        if self.bc[2, 1] == TEMPERATURE:
            tbound = self.ttop
        else:
            tbound = np.array(0.0)
        thot = self.thot
        if T4:
            thot = quad(thot)
            tbound = quad(tbound)
        ttop = np.full((self.nx, self.ny), tbound, dtype=np.float)
        ttop[ihot] = thot
        return ttop

    def copy_plot_data(self):
        new = self.__class__(initial=None)
        for x in ('xf', 'zf', 'rhoc', 'T', 'r', 'geo'):
            setattr(new, x, getattr(self, x))
        return new

    def rho_ini(self, z):
        """
        [from Adelle]
        This function returns the density. If self.rhoflag = deg, then it returns the density for a completely degenerate equation of state. If self.rhoflag = 'fit' it returns a density fit taken from fitting the rho(self.z) profile for a 1D model from Bildsten 1998 written by Frank. If self.rhoflag = 'const', it returns constant density of rho=1e5.
        (profile also agrees with Andrew Cumming's settle code)
        """
        rho0 = np.exp(1.253e1)
        beta = -4.85e-10
        alpha = 4.435
        rho_t = 10
        if self.rhoflag == "deg":
            rho = cube(
                rho_t
                * (1 + (self.g / (4 * self.kb * rho_t ** (1 / 3))) * (self.lz - z))
            )
        elif self.rhoflag == "fit":
            rho = rho0 * np.exp(beta * z ** alpha)
        else:
            rho = rhoflag
        return rho

    def func(self, T_, dt=None, T4=True):
        T = T_.reshape(self.shape)

        i0 = slice(None, -1)
        i1 = slice(1, None)
        ii = slice(1, -1)
        ia = slice(None, None)
        ib = [0, -1]
        im = slice(None, 1)
        ip = slice(-1, None)

        iciaa = (ii, ia, ia)
        icaia = (ia, ii, ia)
        icaai = (ia, ia, ii)

        icx = iciaa
        icy = icaia
        icz = icaai

        ic0aa = (i0, ia, ia)
        ica0a = (ia, i0, ia)
        icaa0 = (ia, ia, i0)

        ic1aa = (i1, ia, ia)
        ica1a = (ia, i1, ia)
        icaa1 = (ia, ia, i1)

        ib0aa = (im, ia, ia)
        ib1aa = (ip, ia, ia)
        iba0a = (ia, im, ia)
        iba1a = (ia, ip, ia)
        ibaa0 = (ia, ia, im)
        ibaa1 = (ia, ia, ip)

        ibbaa = (ib, ia, ia)
        ibaba = (ia, ib, ia)
        ibaab = (ia, ia, ib)

        Tfx = (T[ic1aa] + T[ic0aa]) * 0.5
        Tfy = (T[ica1a] + T[ica0a]) * 0.5
        Tfz = (T[icaa1] + T[icaa0]) * 0.5
        if not T4:
            Tfx3 = cube(Tfx)
            Tfy3 = cube(Tfy)
            Tfz3 = cube(Tfz)

        sfx = (self.nx1, self.ny, self.nz)
        sfy = (self.nx, self.ny1, self.nz)
        sfz = (self.nx, self.ny, self.nz1)

        Fx = np.full(sfx, np.nan)
        Fy = np.full(sfy, np.nan)
        Fz = np.full(sfz, np.nan)

        ktx = np.full(sfx, np.nan)
        kty = np.full(sfy, np.nan)
        ktz = np.full(sfz, np.nan)

        dTfx = np.full(sfx, np.nan)
        dTfy = np.full(sfy, np.nan)
        dTfz = np.full(sfz, np.nan)

        dTfx[iciaa] = T[ic1aa] - T[ic0aa]
        dTfy[icaia] = T[ica1a] - T[ica0a]
        dTfz[icaai] = T[icaa1] - T[icaa0]

        if T4:
            cst = self.cst4
        else:
            cst = self.cst

        kix, kixt = self.kappai_func(Tfx, self.rhox[iciaa], T4=T4)
        kiy, kiyt = self.kappai_func(Tfy, self.rhoy[icaia], T4=T4)
        kiz, kizt = self.kappai_func(Tfz, self.rhoz[icaai], T4=T4)

        cfx = cst * self.rhoxi * self.ax * self.dxfi
        cfy = cst * self.rhoyi * self.ay * self.dyfi
        cfz = cst * self.rhozi * self.az * self.dzfi

        ktx[iciaa] = kixt
        kty[icaia] = kiyt
        ktz[icaai] = kizt

        if not T4:
            dTfx[iciaa] *= Tfx3
            dTfy[icaia] *= Tfy3
            dTfz[icaai] *= Tfz3

        cfx[iciaa] *= kix
        cfy[icaia] *= kiy
        cfz[icaai] *= kiz

        Fx[iciaa] = cfx[iciaa] * dTfx[iciaa]
        Fy[icaia] = cfy[icaia] * dTfy[icaia]
        Fz[icaai] = cfz[icaai] * dTfz[icaai]

        if T4:
            eps, deps = self.net.epsdt4_T4(T, self.rhoc, dt)
        else:
            eps, deps = self.net.epsdt_T(T, self.rhoc, dt)

        S = eps * self.m

        # boundary conditions
        if self.bc[0][0] == REFLECTIVE:
            assert self.bc[0][1] == REFLECTIVE
            Fx[ibbaa] = 0
        elif self.bc[0][0] == PERIODIC:
            assert self.bc[0][1] == PERIODIC
            Tfxb = (T[ib0aa] + T[ib1aa]) * 0.5
            kixb, kixtb = self.kappai_func(Tfxb, self.rhox[ib0aa], T4=T4)
            cfx[ibbaa] *= kixb
            ktx[ibbaa] = kixtb
            dTfxb = T[ib0aa] - T[ib1aa]
            if not T4:
                dTfxb *= cube(Tfxb)
            Fx[ibbaa] = cfx[ib0aa] * dTfxb
            dTfx[ibbaa] = dTfxb
        else:
            raise AttributeError(
                f'Invalid boundary condition {self.bc[0][0]} in x-direction.'
            )

        if self.bc[1][0] == REFLECTIVE:
            assert self.bc[1][1] == REFLECTIVE
            Fy[ibaba] = 0
        elif self.bc[1][0] == PERIODIC:
            assert self.bc[1][1] == PERIODIC
            Tfyb = (T[iba0a] + T[iba1a]) * 0.5
            kiyb, kiytb = self.kappai_func(Tfyb, self.rhoy[iba1a], T4=T4)
            cfy[ibaba] *= kiyb
            kty[ibaba] = kiytb
            dTfyb = T[iba0a] - T[iba1a]
            if not T4:
                dTfyb *= cube(Tfyb)
            Fy[ibaba] = cfy[iba0a] * dTfyb
            dTfy[ibaba] = dTfyb
        else:
            raise AttributeError(
                f'Invalid boundary condition {self.bc[1][0]} in y-direction.'
            )

        if self.bc[2][0] == FLUX:
            Fz[ibaa0] = self.fbase * self.az[ibaa0]
        elif self.bc[2][0] == TEMPERATURE:
            if T4:
                tbase = np.tile(quad(self.tbase), (self.nx, self.ny, 1))
            else:
                tbase = np.tile(self.tbase, (self.nx, self.ny, 1))
            Tfzb0 = (T[ibaa0] + tbase) * 0.5
            dTfzb0 = T[ibaa0] - tbase
            if not T4:
                dTfzb0 *= cube(Tfzb0)
            kizb0, kiztb0 = self.kappai_func(Tfzb0, self.rhoz[ibaa0], T4=T4)
            cfz[ibaa0] *= kizb0
            ktz[ibaa0] = kiztb0
            Fz[ibaa0] = cfz[ibaa0] * dTfzb0
            dTfz[ibaa0] = dTfzb0
        else:
            raise AttributeError(f'Invalid boundary condition {self.bc[2][o]} at base.')

        assert self.bc[2][1] in (TEMPERATURE, BLACKBODY,)
        ttop = self.ttop_func(T4)[:, :, np.newaxis]
        Tfzb1 = (T[ibaa1] + ttop) * 0.5
        if self.bc[2][1] == BLACKBODY:
            iibb = ttop == 0
            cfz[ibaa1][iibb] = (-self.SB) * self.az[ibaa1][iibb]
            ktz[ibaa1][iibb] = 0

            iihs = ~iibb
            kizb1, kiztb1 = self.kappai_func(Tfzb1[iihs], self.rhoz[ibaa1][iihs], T4=T4)
            cfz[ibaa1][iihs] *= kizb1
            ktz[ibaa1][iihs] = kiztb1
        elif self.bc[2][1] == TEMPERATURE:
            kizb1, kiztb1 = self.kappai_func(Tfzb1, self.rhoz[ibaa1], T4=T4)
            cfz[ibaa1] *= kizb1
            ktz[ibaa1] = kiztb1
        else:
            raise AttributeError(
                f'Invalid boundary condition {self.bc[2][1]} at surface.'
            )
        if T4:
            dTfzb1 = ttop - T[ibaa1]
        else:
            dTfzb1 = (ttop - T[ibaa1]) * cube(Tfzb1)
        Fz[ibaa1] = cfz[ibaa1] * dTfzb1
        dTfz[ibaa1] = dTfzb1

        # sum all terms
        F = (
            S + Fx[ic0aa] - Fx[ic1aa] + Fy[ica0a] - Fy[ica1a] + Fz[icaa0] - Fz[icaa1]
        ).flatten()

        if not self.jac:
            return F

        # ====================
        #       Jacobian
        # ====================
        if self.bc[0][0] == REFLECTIVE:
            cfx[ibbaa] = 0
            ktx[ibbaa] = 0
            dTfx[ibbaa] = 0
        if self.bc[1][0] == REFLECTIVE:
            cfy[ibaba] = 0
            kty[ibaba] = 0
            dTfy[ibaba] = 0
        if self.bc[2][0] == FLUX:
            cfz[ibaa0] = 0
            ktz[ibaa0] = 0
            dTfz[ibaa0] = 0

        if not hasattr(self, '_s'):
            diag = np.mgrid[0 : self.nx, 0 : self.ny, 0 : self.nz]
            i00 = tuple(np.r_[diag, diag])
            imx = (
                diag[0][ic1aa],
                diag[1][ic1aa],
                diag[2][ic1aa],
                diag[0][ic1aa] - 1,
                diag[1][ic1aa],
                diag[2][ic1aa],
            )
            ipx = (
                diag[0][ic0aa],
                diag[1][ic0aa],
                diag[2][ic0aa],
                diag[0][ic0aa] + 1,
                diag[1][ic0aa],
                diag[2][ic0aa],
            )
            imy = (
                diag[0][ica1a],
                diag[1][ica1a],
                diag[2][ica1a],
                diag[0][ica1a],
                diag[1][ica1a] - 1,
                diag[2][ica1a],
            )
            ipy = (
                diag[0][ica0a],
                diag[1][ica0a],
                diag[2][ica0a],
                diag[0][ica0a],
                diag[1][ica0a] + 1,
                diag[2][ica0a],
            )
            imz = (
                diag[0][icaa1],
                diag[1][icaa1],
                diag[2][icaa1],
                diag[0][icaa1],
                diag[1][icaa1],
                diag[2][icaa1] - 1,
            )
            ipz = (
                diag[0][icaa0],
                diag[1][icaa0],
                diag[2][icaa0],
                diag[0][icaa0],
                diag[1][icaa0],
                diag[2][icaa0] + 1,
            )

            if self.bc[0][0] == PERIODIC:
                ix0 = (
                    diag[0][ib0aa],
                    diag[1][ib0aa],
                    diag[2][ib0aa],
                    diag[0][ib1aa],
                    diag[1][ib1aa],
                    diag[2][ib1aa],
                )
                ix1 = (
                    diag[0][ib1aa],
                    diag[1][ib1aa],
                    diag[2][ib1aa],
                    diag[0][ib0aa],
                    diag[1][ib0aa],
                    diag[2][ib0aa],
                )
            if self.bc[1][0] == PERIODIC:
                iy0 = (
                    diag[0][iba0a],
                    diag[1][iba0a],
                    diag[2][iba0a],
                    diag[0][iba1a],
                    diag[1][iba1a],
                    diag[2][iba1a],
                )
                iy1 = (
                    diag[0][iba1a],
                    diag[1][iba1a],
                    diag[2][iba1a],
                    diag[0][iba0a],
                    diag[1][iba0a],
                    diag[2][iba0a],
                )

            nn = self.nx * self.ny * self.nz
            if self.sparse:
                m = np.arange(nn).reshape(self.shape)

                def s_(ii):
                    return (m[ii[:3]].flatten(), m[ii[3:]].flatten())

                i00 = s_(i00)
                imx = s_(imx)
                ipx = s_(ipx)
                imy = s_(imy)
                ipy = s_(ipy)
                imz = s_(imz)
                ipz = s_(ipz)
                s = -1
                if self.bc[0][0] == PERIODIC:
                    ix0 = s_(ix0)
                    ix1 = s_(ix1)
                if self.bc[1][0] == PERIODIC:
                    iy0 = s_(iy0)
                    iy1 = s_(iy1)
            else:
                s = None
            _s = _Diffuse_Store()
            _s.s = s
            _s.i00 = i00
            _s.ipx = ipx
            _s.imx = imx
            _s.ipy = ipy
            _s.imy = imy
            _s.ipz = ipz
            _s.imz = imz
            _s.nn = nn
            self._s = _s
            if self.bc[0][0] == PERIODIC:
                _s.ix0 = ix0
                _s.ix1 = ix1
            if self.bc[1][0] == PERIODIC:
                _s.iy0 = iy0
                _s.iy1 = iy1
        else:
            _s = self._s
            s = _s.s
            nn = _s.nn
            i00 = _s.i00
            ipx = _s.ipx
            imx = _s.imx
            ipy = _s.ipy
            imy = _s.imy
            ipz = _s.ipz
            imz = _s.imz
            if self.bc[0][0] == PERIODIC:
                ix0 = _s.ix0
                ix1 = _s.ix1
            if self.bc[1][0] == PERIODIC:
                iy0 = _s.iy0
                iy1 = _s.iy1

        if self.sparse:
            dF = lil_matrix((nn, nn))
        else:
            dF = np.zeros(self.shape * 2)

        if T4:
            if self.const_kappa:
                dF[i00] = (
                    deps * self.m
                    + cfx[ic0aa]
                    + cfx[ic1aa]
                    + cfy[ica0a]
                    + cfy[ica1a]
                    + cfz[icaa0]
                    + cfz[icaa1]
                ).reshape(s)
                dF[imx] = -cfx[icx].reshape(s)
                dF[ipx] = -cfx[icx].reshape(s)
                dF[ipy] = -cfy[icy].reshape(s)
                dF[imy] = -cfy[icy].reshape(s)
                dF[imz] = -cfz[icz].reshape(s)
                dF[ipz] = -cfz[icz].reshape(s)
                if self.bc[0][0] == PERIODIC:
                    # it should be the case that cfx[ib0aa] == cfx[ib1aa]
                    dF[ix0] = -cfx[ib0aa].reshape(s)
                    dF[ix1] = -cfx[ib0aa].reshape(s)
                if self.bc[1][0] == PERIODIC:
                    # it should be the case that cfy[iba0a] == cfy[iba1a]
                    dF[iy0] = -cfy[iba0a].reshape(s)
                    dF[iy1] = -cfy[iba0a].reshape(s)
            else:
                dF[i00] = (
                    deps * self.m
                    + cfx[ic0aa] * (1 + dTfx[ic0aa] * ktx[ic0aa])
                    + cfx[ic1aa] * (1 - dTfx[ic1aa] * ktx[ic1aa])
                    + cfy[ica0a] * (1 + dTfy[ica0a] * kty[ica0a])
                    + cfy[ica1a] * (1 - dTfy[ica1a] * kty[ica1a])
                    + cfz[icaa0] * (1 + dTfz[icaa0] * ktz[icaa0])
                    + cfz[icaa1] * (1 - dTfz[icaa1] * ktz[icaa1])
                ).reshape(s)
                dF[imx] = (cfx[icx] * (-1 + dTfx[icx] * ktx[icx])).reshape(s)
                dF[ipx] = (cfx[icx] * (-1 - dTfx[icx] * ktx[icx])).reshape(s)
                dF[ipy] = (cfy[icy] * (-1 + dTfy[icy] * kty[icy])).reshape(s)
                dF[imy] = (cfy[icy] * (-1 - dTfy[icy] * kty[icy])).reshape(s)
                dF[imz] = (cfz[icz] * (-1 + dTfz[icz] * ktz[icz])).reshape(s)
                dF[ipz] = (cfz[icz] * (-1 - dTfz[icz] * ktz[icz])).reshape(s)
                if self.bc[0][0] == PERIODIC:
                    # it should be the case that cfx[ib0aa] == cfx[ib1aa], same for ktx
                    dF[ix0] = (cfx[ib0aa] * (-1 + dTfx[ib0aa] * ktx[ib0aa])).reshape(s)
                    dF[ix1] = (cfx[ib1aa] * (-1 - dTfx[ib1aa] * ktx[ib1aa])).reshape(s)
                if self.bc[1][0] == PERIODIC:
                    # it should be the case that cfy[iba0a] == cfy[iba1a], same for kty
                    dF[iy0] = (cfy[iba0a] * (-1 + dTfy[iba0a] * kty[iba0a])).reshape(s)
                    dF[iy1] = (cfy[iba1a] * (-1 - dTfy[iba1a] * kty[iba1a])).reshape(s)
        else:
            T2 = T ** 2
            T3 = T2 * T
            T32 = T3 * 0.25
            T34 = T3 * 0.5
            T26 = T2 * 0.75

            if self.bc[0][0] == REFLECTIVE:
                Tmx = np.vstack((np.zeros((1, self.ny, self.nz)), T[ic0aa]))
                Tmx32 = np.vstack((np.zeros((1, self.ny, self.nz)), T32[ic0aa]))
                Tpx = np.vstack((T[ic1aa], np.zeros((1, self.ny, self.nz))))
                Tpx32 = np.vstack((T32[ic1aa], np.zeros((1, self.ny, self.nz))))
            else:
                Tmx = np.roll(T, 1, axis=0)
                Tmx32 = np.roll(T32, 1, axis=0)
                Tpx = np.roll(T, -1, axis=0)
                Tpx32 = np.roll(T32, -1, axis=0)

            if self.bc[1][0] == REFLECTIVE:
                Tmy = np.hstack((np.zeros((self.nx, 1, self.nz)), T[ica0a]))
                Tmy32 = np.hstack((np.zeros((self.nx, 1, self.nz)), T32[ica0a]))
                Tpy = np.hstack((T[ica1a], np.zeros((self.nx, 1, self.nz))))
                Tpy32 = np.hstack((T32[ica1a], np.zeros((self.nx, 1, self.nz))))
            else:
                Tmy = np.roll(T, 1, axis=1)
                Tmy32 = np.roll(T32, 1, axis=1)
                Tpy = np.roll(T, -1, axis=1)
                Tpy32 = np.roll(T32, -1, axis=1)

            if self.bc[2][0] == FLUX:
                Tmz = np.dstack((np.zeros((self.nx, self.ny, 1)), T[icaa0]))
                Tmz32 = np.dstack((np.zeros((self.nx, self.ny, 1)), T32[icaa0]))
            else:
                Tmz = np.dstack((tbase.reshape(self.nx, self.ny, 1), T[icaa1],))
                Tmz32 = np.dstack(
                    ((0.25 * cube(ttop)).reshape(self.nx, self.ny, 1), T32[icaa1],)
                )
            # add TEMEPRATURE CASE?
            Tpz = np.dstack((T[icaa1], ttop.reshape(self.nx, self.ny, 1)))
            Tpz32 = np.dstack(
                (T32[icaa1], (0.25 * cube(ttop)).reshape(self.nx, self.ny, 1))
            )

            if self.const_kappa:
                dF[i00] = (
                    deps * self.m
                    + cfx[ic0aa] * (T34 + T26 * Tmx - Tmx32)
                    + cfx[ic1aa] * (T34 + T26 * Tpx - Tpx32)
                    + cfy[ica0a] * (T34 + T26 * Tmy - Tmy32)
                    + cfy[ica1a] * (T34 + T26 * Tpy - Tpy32)
                    + cfz[icaa0] * (T34 + T26 * Tmz - Tmz32)
                    + cfz[icaa1] * (T34 + T26 * Tpz - Tpz32)
                ).reshape(s)
                dF[imx] = (
                    cfx[icx] * (T32[ic1aa] - T[ic1aa] * T26[ic0aa] - T34[ic0aa])
                ).reshape(s)
                dF[ipx] = (
                    cfx[icx] * (T32[ic0aa] - T[ic0aa] * T26[ic1aa] - T34[ic1aa])
                ).reshape(s)
                dF[imy] = (
                    cfy[icy] * (T32[ica1a] - T[ica1a] * T26[ica0a] - T34[ica0a])
                ).reshape(s)
                dF[ipy] = (
                    cfy[icy] * (T32[ica0a] - T[ica0a] * T26[ica1a] - T34[ica1a])
                ).reshape(s)
                dF[imz] = (
                    cfz[icz] * (T32[icaa1] - T[icaa1] * T26[icaa0] - T34[icaa0])
                ).reshape(s)
                dF[ipz] = (
                    cfz[icz] * (T32[icaa0] - T[icaa0] * T26[icaa1] - T34[icaa1])
                ).reshape(s)

                if self.bc[0][0] == PERIODIC:
                    # it should be the case that cfx[ib0aa] == cfx[ib1aa]
                    dF[ix0] = (
                        cfx[ib0aa] * (T32[ib0aa] - T[ib0aa] * T26[ib1aa] - T34[ib1aa])
                    ).reshape(s)
                    dF[ix1] = (
                        cfx[ib1aa] * (T32[ib1aa] - T[ib1aa] * T26[ib0aa] - T34[ib0aa])
                    ).reshape(s)
                if self.bc[1][0] == PERIODIC:
                    # it should be the case that cfy[iba0a] == cfy[iba1a]
                    dF[iy0] = (
                        cfy[iba0a] * (T32[iba0a] - T[iba0a] * T26[iba1a] - T34[iba1a])
                    ).reshape(s)
                    dF[iy1] = (
                        cfy[iba1a] * (T32[iba1a] - T[iba1a] * T26[iba0a] - T34[iba0a])
                    ).reshape(s)
            else:
                dF[i00] = (
                    deps * self.m
                    + cfx[ic0aa] * (T34 + T26 * Tmx - Tmx32 + dTfx[ic0aa] * ktx[ic0aa])
                    + cfx[ic1aa] * (T34 + T26 * Tpx - Tpx32 - dTfx[ic1aa] * ktx[ic1aa])
                    + cfy[ica0a] * (T34 + T26 * Tmy - Tmy32 + dTfy[ica0a] * kty[ica0a])
                    + cfy[ica1a] * (T34 + T26 * Tpy - Tpy32 - dTfy[ica1a] * kty[ica1a])
                    + cfz[icaa0] * (T34 + T26 * Tmz - Tmz32 + dTfz[icaa0] * ktz[icaa0])
                    + cfz[icaa1] * (T34 + T26 * Tpz - Tpz32 - dTfz[icaa1] * ktz[icaa1])
                ).reshape(s)
                dF[imx] = (
                    cfx[icx]
                    * (
                        T32[ic1aa]
                        - T[ic1aa] * T26[ic0aa]
                        - T34[ic0aa]
                        + dTfx[icx] * ktx[icx]
                    )
                ).reshape(s)
                dF[ipx] = (
                    cfx[icx]
                    * (
                        T32[ic0aa]
                        - T[ic0aa] * T26[ic1aa]
                        - T34[ic1aa]
                        - dTfx[icx] * ktx[icx]
                    )
                ).reshape(s)
                dF[imy] = (
                    cfy[icy]
                    * (
                        T32[ica1a]
                        - T[ica1a] * T26[ica0a]
                        - T34[ica0a]
                        + dTfy[icy] * kty[icy]
                    )
                ).reshape(s)
                dF[ipy] = (
                    cfy[icy]
                    * (
                        T32[ica0a]
                        - T[ica0a] * T26[ica1a]
                        - T34[ica1a]
                        - dTfy[icy] * kty[icy]
                    )
                ).reshape(s)
                dF[imz] = (
                    cfz[icz]
                    * (
                        T32[icaa1]
                        - T[icaa1] * T26[icaa0]
                        - T34[icaa0]
                        + dTfz[icz] * ktz[icz]
                    )
                ).reshape(s)
                dF[ipz] = (
                    cfz[icz]
                    * (
                        T32[icaa0]
                        - T[icaa0] * T26[icaa1]
                        - T34[icaa1]
                        - dTfz[icz] * ktz[icz]
                    )
                ).reshape(s)

                if self.bc[0][0] == PERIODIC:
                    # it should be the case that cfx[ib0aa] == cfx[ib1aa]
                    dF[ix0] = (
                        cfx[ib0aa]
                        * (
                            T32[ib0aa]
                            - T[ib0aa] * T26[ib1aa]
                            - T34[ib1aa]
                            + dTfx[ib0aa] * ktx[ib0aa]
                        )
                    ).reshape(s)
                    dF[ix1] = (
                        cfx[ib1aa]
                        * (
                            T32[ib1aa]
                            - T[ib1aa] * T26[ib0aa]
                            - T34[ib0aa]
                            - dTfx[ib1aa] * ktx[ib1aa]
                        )
                    ).reshape(s)
                if self.bc[1][0] == PERIODIC:
                    # it should be the case that cfy[iba0a] == cfy[iba1a]
                    dF[iy0] = (
                        cfy[iba0a]
                        * (
                            T32[iba0a]
                            - T[iba0a] * T26[iba1a]
                            - T34[iba1a]
                            + dTfy[iba0a] * kty[iba0a]
                        )
                    ).reshape(s)
                    dF[iy1] = (
                        cfy[iba1a]
                        * (
                            T32[iba1a]
                            - T[iba1a] * T26[iba0a]
                            - T34[iba0a]
                            - dTfy[iba1a] * kty[iba1a]
                        )
                    ).reshape(s)

        if self.sparse:
            dF = dF.tocsc()
        else:
            dF = dF.reshape((nn, nn))
        return F, dF

    def solve(self, T, op=None, tol=1e-6, f=None, lofac=1e-3):
        if f is None:
            f = self.func
        if self.sparse:
            solver = partial(spsolve, use_umfpack=False)
        else:
            solver = solve
        t = np.ndarray((4, 3))
        dt = np.zeros((3, 3))
        for i in range(1000):
            self.iter = i
            use = resource.getrusage(resource.RUSAGE_SELF)
            t[0, :] = (time.time(), use.ru_utime, use.ru_stime)
            b, A = f(T)
            use = resource.getrusage(resource.RUSAGE_SELF)
            t[1, :] = (time.time(), use.ru_utime, use.ru_stime)
            c = solver(A, b).reshape(self.shape)
            use = resource.getrusage(resource.RUSAGE_SELF)
            t[2, :] = (time.time(), use.ru_utime, use.ru_stime)

            ierr = np.unravel_index(np.argmax(np.abs(c / T)), self.shape)
            err = c[ierr] / T[ierr]
            if op is not None:
                opx = np.minimum(1, op / np.abs(err))
            else:
                opx = 1
            ierr0 = np.unravel_index(np.argmax(c / T), self.shape)
            err0 = c[ierr0] / T[ierr0]
            if err0 > 1 - lofac:
                opx = np.minimum(opx, (1 - lofac) / (1 + err0))
            if opx != 1:
                T -= c * opx
            else:
                T -= c
            use = resource.getrusage(resource.RUSAGE_SELF)
            t[3, :] = (time.time(), use.ru_utime, use.ru_stime)
            dt[:, :] += t[1:, :] - t[:-1, :]
            Terr = T[ierr]
            if f.keywords.get('T4', True):
                Terr = qqrt(Terr)
            print(
                f'[DIFFUSE.SOLVE] {i:03d}: err = {err:+8e}, T = {Terr:+8e}, opx = {opx:+8e}, ierr = {ierr}'
            )
            if np.isnan(err):
                raise Exception('No convergence')
            if np.abs(err) < tol:
                break
        else:
            raise Exception('No convergence')
        dt = np.hstack((dt, np.sum(dt[:, 1:], axis=1)[:, np.newaxis]))
        dt = np.vstack((dt, np.sum(dt[:, :], axis=0)[np.newaxis, :]))
        for m, x in zip(['setup', 'solve', 'update', 'total'], dt):
            print(
                f'[DIFFUSE.SOLVE] {m:<6s}  time: {time2human(x[0]):>8s} real, {time2human(x[1]):>8s} user, {time2human(x[2]):>8s} sys, {time2human(x[3]):>8s} cpu total.'
            )
        result = _Diffuse_Result()
        result.x = T
        result.nit = i + 1
        result.message = 'converged'
        result.success = True
        return result

    def init_store(self):
        self.store = list()

    def add_store(self):
        self.store.append(dict(T=self.T.copy(), t=self.t, net=copy.deepcopy(self.net)))

    def save_store(self, filename):
        filename = Path(filename).expanduser()
        with open(filename, 'wb') as f:
            pickle.dump(self.store, f)

    def load_store(self, filename):
        filename = Path(filename).expanduser()
        with open(filename, 'rb') as f:
            self.store = picke.load(f)

    def save(self, filename=None):
        if filename is None:
            if not hasattr(self, 'uuid'):
                self.uuid = uuid1()
            filename = f'{self.uuid.hex}.pickle.xz'
        filename = Path(filename).expanduser()
        if filename.suffix == '.gz':
            f = gzip.open(filename, 'wb')
        elif filename.suffix == '.bz':
            f = bz2.BZ2File(filename, 'wb')
        elif filename.suffix == '.xz':
            f = lzma.LZMAFile(filename, 'wb')
        else:
            f = open(filename, 'wb')
        pickle.dump(self, f)
        f.close()

    @classmethod
    def load(cls, filename):
        filename = Path(filename).expanduser()
        if filename.suffix == '.gz':
            f = gzip.open(filename, 'rb')
        elif filename.suffix == '.bz':
            f = bz2.BZ2File(filename, 'rb')
        elif filename.suffix == '.xz':
            f = lzma.LZMAFile(filename, 'rb')
        else:
            f = open(filename, 'rb')
        self = pickle.load(f)
        f.close()
        if not isinstance(self, cls):
            raise AttributeError(f'"{filename}" is wrong type.')
        print(
            f'[DIFFUSE.LOAD] geo={self.geo}, nx={self.nx}, ny={self.ny}, nz={self.nz}.'
        )
        return self

    def evo_step(self, T, dT=None, dt=1e-4, t=None, maxit=20):
        """
        Henyey-style solver (implicit)
        """
        T0 = T.flatten()
        T0i = 1 / T0

        dti = 1 / dt
        Cv0 = self.Cv.flatten() * dti
        Cv1 = diags(Cv0, format='csc')

        if self.method == 'solve4':
            f = partial(self.func, T4=True)
            T0q = qqrt(T0)
            Cv04 = Cv0 * 0.25
        else:
            f = partial(self.func, T4=False)
        solver = spsolve

        if dT is None:
            dT = np.zeros_like(T0)
        else:
            dT = dT.flatten()

        Tn = T0 + dT
        for i in range(maxit):
            F, dF = f(Tn, dt)
            if self.method == 'solve4':
                Tnq = qqrt(Tn)
                b = F - Cv0 * (Tnq - T0q)
                A = dF - diags(Cv04 * Tnq / Tn, format='csc')
            else:
                b = F - Cv0 * dT
                A = dF - Cv1
            c = solver(A, b)
            Tn = Tn - c
            dT = Tn - T0
            dTs = np.max(np.abs(c) * T0i)
            dTr = np.max(np.abs(dT) * T0i)
            print(f'[EVO_STEP] [{i:04}] dTr = {dTr:+8e}, dTs = {dTs:+8e}')
            if dTs < 1e-6:
                break
        else:
            return None, 0, 0
        Tn = Tn.reshape(self.shape)
        return Tn, dT, dTr

    def evo(
        self,
        tstop=1000,
        nstep=None,
        dt0=None,
        dTr=None,
        movie=None,
        mkwargs=dict(),
        pkwargs=dict(),
        fkwargs=dict(),
        dtmax=None,
        store=True,
        t=None,
        dtf=0.5 * (np.sqrt(5) + 1),
    ):
        if t is None:
            t = getattr(self, 't', 0)
        self.t = t
        if dt0 is None:
            dt0 = getattr(self, 'dt', 1e-4)
        dt = dt0
        dT = getattr(self, 'dT', None)
        xt = np.ndarray((5, 3))
        use = resource.getrusage(resource.RUSAGE_SELF)
        xt[:, :] = np.array((time.time(), use.ru_utime, use.ru_stime))[np.newaxis, :]
        if movie is not None:
            fkw = dict(size=(800, 320))
            fkw = dict(fkw, **fkwargs)
            mkw = dict(delay=0.1)
            mkw = dict(mkw, **mkwargs)
            pkw = dict(q='T', scale='log')
            pkw = dict(pkw, **pkwargs)
            q = pkw.pop('q', 'T')
            movie = MovieWriter(
                movie,
                canvas=MPLCanvas(**fkw),
                generator=self.plot,
                gargs=(q,),
                gkwargs=pkw,
                **mkw,
            )
            movie.write(gkwargs=dict(t=t))
            use = resource.getrusage(resource.RUSAGE_SELF)
            xt[3, :] = (time.time(), use.ru_utime, use.ru_stime)
            xm = xt[3] - xt[2]
        if store:
            if t == 0:
                self.init_store()
            self.add_store()
        T = self.T
        if self.method == 'solve4':
            T = quad(T)
            if dTr is None:
                dTr = 1
        if dTr is None:
            dTr = 0.1
        Tn = T
        if dtmax is None:
            dtmax = tstop * 0.1
        i = getattr(self, 'i', 0)
        if nstep is not None:
            nstop = i + nstep
            tstop = 1e30
        else:
            nstop = 2 ** 30
        while (t < tstop) and (i < nstop):
            xt[1] = xt[2]
            i += 1
            Tp = T
            T = Tn
            dT0 = dT
            Tn, dT, dTr_ = self.evo_step(T, dT=dT, dt=dt, t=t)
            if Tn is None or dTr_ > 2 * dTr:
                print(f'[EVO] backup')
                dt = dt / 10
                i = i - 1
                dT = dT0
                Tn = T
                T = Tp
                continue
            if self.method == 'solve4':
                self.T = qqrt(Tn)
            else:
                self.T = Tn
            self.eps = self.net.epsdt(self.T, self.rhoc, dt, update=True)
            dtn = min((dtmax, dt * dTr / dTr_, dtf * dt, 0.1 * self.net.dt))
            # should backup if new time step is a lot less than dt
            if dtn < dt / dtf:
                print(f'[EVO] backup dtn')
                dt = dt / 10
                i = i - 1
                dT = dT0
                Tn = T
                T = Tp
                self.net.revert_update()
                continue
            t += dt
            self.t = t
            dt0 = dt
            dt = dtn
            dT = dT * (dt / dt0)
            use = resource.getrusage(resource.RUSAGE_SELF)
            xt[2, :] = (time.time(), use.ru_utime, use.ru_stime)
            x = xt[2] - xt[1]
            x = np.hstack((x, np.sum(x[1:])))
            print(
                f'[EVO] [{i:06d}] time = {t:+8e}, dtn = {dt:+8e}, real: {time2human(x[0]):>8s}, CPU: {time2human(x[3]):>8s}'
            )
            if movie is not None:
                movie.write(gkwargs=dict(t=t))
                use = resource.getrusage(resource.RUSAGE_SELF)
                xt[3, :] = (time.time(), use.ru_utime, use.ru_stime)
                xm += xt[3] - xt[2]
            if store:
                self.add_store()
        if self.method == 'solve4':
            Tn = qqrt(Tn)
        self.T = Tn
        self.dt = dt
        self.dT = dT
        self.i = i
        if movie is not None:
            use = resource.getrusage(resource.RUSAGE_SELF)
            xt[2, :] = (time.time(), use.ru_utime, use.ru_stime)
            movie.close()
            use = resource.getrusage(resource.RUSAGE_SELF)
            xt[3, :] = (time.time(), use.ru_utime, use.ru_stime)
            xm += xt[3] - xt[2]
            x = np.hstack((xm, np.sum(xm[1:])))
            print(
                f'[EVO] movie:   time: {time2human(x[0]):>8s} real, {time2human(x[1]):>8s} user, {time2human(x[2]):>8s} sys, {time2human(x[3]):>8s} cpu total.'
            )
        use = resource.getrusage(resource.RUSAGE_SELF)
        xt[-1, :] = (time.time(), use.ru_utime, use.ru_stime)
        x = xt[-1] - xt[0]
        x = np.hstack((x, np.sum(x[1:])))
        print(
            f'[EVO] total:   time: {time2human(x[0]):>8s} real, {time2human(x[1]):>8s} user, {time2human(x[2]):>8s} sys, {time2human(x[3]):>8s} cpu total.'
        )

    def store2movie(
        self,
        movie=None,
        mkwargs=dict(),
        pkwargs=dict(),
        fkwargs=dict(),
        sel=slice(None),
        mp=True,
    ):
        xt = np.ndarray((2, 3))
        use = resource.getrusage(resource.RUSAGE_SELF)
        usc = resource.getrusage(resource.RUSAGE_CHILDREN)
        xt[:, :] = np.array(
            (time.time(), use.ru_utime + usc.ru_utime, use.ru_stime + usc.ru_stime,)
        )[np.newaxis, :]
        fkw = dict(size=(800, 400))
        fkw = dict(fkw, **fkwargs)
        mkw = dict(delay=0.1, mp=mp)
        mkw = dict(mkw, **mkwargs)
        pkw = dict(q=('eps', 'T'), plot='plot2')
        pkw = dict(pkw, **pkwargs)
        plot = pkw.pop('plot')
        q = pkw.pop('q')
        plotter = getattr(self.copy_plot_data(), plot)
        make_movie(
            movie,
            generator=plotter,
            canvas=MPLCanvas,
            ckwargs=fkw,
            gargs=(q,),
            gkwargs=pkw,
            values=self.store[sel],
            data='kwargs',
            **mkw,
        )
        use = resource.getrusage(resource.RUSAGE_SELF)
        usc = resource.getrusage(resource.RUSAGE_CHILDREN)
        xt[1, :] = np.array(
            (time.time(), use.ru_utime + usc.ru_utime, use.ru_stime + usc.ru_stime,)
        )[np.newaxis, :]
        xm = xt[1] - xt[0]
        x = np.hstack((xm, np.sum(xm[1:])))
        print(
            f'[EVO] movie:   time: {time2human(x[0]):>8s} real, {time2human(x[1]):>8s} user, {time2human(x[2]):>8s} sys, {time2human(x[3]):>8s} cpu total.'
        )

    def Q(self, q=None, T=None, net=None):
        if q is None:
            q = 'T'
        if T is None:
            T = self.T.copy()
        if net is None:
            net = self.net
        cmap_ = None
        cmap = None
        if q == 'eps':
            Q = net.eps(T, self.rhoc) + TINY
            label = r'$\varepsilon_{\mathrm{nuc}}$'
            unit = 'erg/g/s'
            defaultscale = 'log'
            cmap = color.colormap('plasma')
        elif q == 'deps':
            _, Q = net.eps_T(T, self.rhoc)
            label = r'$\frac{\mathrm{d}\,\varepsilon_{\mathrm{nuc}}}{\mathrm{d}\,T}$'
            unit = 'erg/g/s/K'
            defaultscale = 'lin'
        elif q == 'dleps':
            e, Q = net.eps_T(T, self.rhoc)
            Q = Q / e * T
            label = r'$\frac{\mathrm{d}\,\ln\,\varepsilon_{\mathrm{nuc}}}{\mathrm{d}\,\ln\,\,T}$'
            unit = None
            defaultscale = 'lin'
        elif q == 'dleps':
            e, Q = net.eps_T(T, self.rhoc)
            Q = Q / e * T
            label = r'$\frac{\mathrm{d}\,\ln\,\varepsilon_{\mathrm{nuc}}}{\mathrm{d}\,\ln\,\,T}$'
            unit = None
            defaultscale = 'lin'
        elif q == 'he4':
            Q = net.massfrac('he4')
            label = r'$^4\mathrm{He}$ mass fraction'
            unit = None
            defaultscale = 'lin'
            cmap_ = color.colormap('bone_r')
        elif q == 'c12':
            Q = net.massfrac('c12')
            label = r'$^{12}\mathrm{C}$ mass fraction'
            unit = None
            defaultscale = 'lin'
            cmap_ = color.colormap('gray_r')
        elif q == 'mg24':
            Q = net.massfrac('mg24')
            label = r'$^{24}\mathrm{Mg}$ mass fraction'
            unit = None
            defaultscale = 'lin'
            cmap_ = color.colormap('pink_r')
        elif q == 'rate':
            Q = net.rate(T, self.rhoc)
            label = 'rate'
            unit = r'$\mathrm{mol}\,\mathrm{g}^{-1}\,\mathrm{s}^{-1}$'
            defaultscale = 'log'
        elif q == 'T':
            Q = T
            label = 'Temperature'
            unit = 'K'
            defaultscale = 'log'
            cmap = color.ColorBlindRainbow()
        elif q == 'rho':
            Q = self.rhoc
            label = 'Density'
            unit = '$\mathrm{g}\,\mathrm{cm}^{-3}$'
            defaultscale = 'log'
            cmap = color.ColorBlindRainbow()
        elif q == 'k':
            ki, kit = self.kappai_func(T, self.rhoc, T4=False)
            Q = 1 / ki
            label = 'opacity'
            unit = r'$\mathrm{cm}^2\,\mathrm{g}^{-1}$'
            defaultscale = 'log'
            cmap = color.colormap('magma_r')
        else:
            raise AttributeError(f'Unknown plot quantity: {q}')
        if cmap_ is not None:
            cmap = cmap_
        else:
            if cmap is None:
                cmap = color.colormap('cividis')
            # cmap = color.WaveFilter(cmap, nwaves = 40, amplitude=0.2)
            cmap = plt.cm.magma
        return Q, label, unit, defaultscale, cmap

    def plot(
        self,
        q='T',
        /,
        scale=None,
        mirror=False,
        fig=None,
        t=None,
        vmin=None,
        vmax=None,
        T=None,
        net=None,
        xlim=None,
        ylim=None,
        zlim=None,
        pos=None,
        cbpos=None,
        ix=None,
        iy=None,
        iz=None,
        flatten=True,
        mode=None,  # voxel, project, coord, shell
    ):
        Q, label, unit, defaultscale, cmap = self.Q(q, T, net)
        ndim = np.squeeze(Q).ndim
        if mode is None:
            if ndim == 2:
                mode = 'project'
            else:
                mode = 'voxel'
        if ix is None and iy is None and iz is None:
            if self.nx == 1:
                ix = -1
            elif self.ny == 1:
                iy = -1
            elif self.nz == 1:
                iz = -1
            elif mode == 'voxel':
                iy = slice(None)
            else:
                iy = -1

        # geometries
        if self.geo in ('cartesian',):
            label3d = ('Horizontal Distance', 'Depths', 'Height')
        elif self.geo in ('cylindrical',):
            label3d = ('Horizontal Distance', 'Circumferrence', 'Height')
        elif self.geo in ('sphericall',):
            label3d = ('Horizontal Distance', 'Circumferrence', 'Radial Distance')
        elif self.geo in ('patch', 'spherical',):
            label3d = (
                'Longitudinal Distance',
                'Latitudinal Distance',
                'Radial Distance',
            )
        else:
            raise AttributeError(f'Geometry "{self.geo}" not supported.')

        if mode == 'voxel':
            coord3d = np.moveaxis(self.vf.copy(), -1, 0)
        elif mode == 'coord':
            coord3d = (self.xf, self.yf, self.zf)
        elif mode in ('shell', 'project'):
            if ix is not None:
                coord3d = np.moveaxis(self.vxe.copy(), -1, 0)
            elif iy is not None:
                coord3d = np.moveaxis(self.vye.copy(), -1, 0)
            else:
                coord3d = np.moveaxis(self.vze.copy(), -1, 0)
            if mode in ('project',):
                if self.geo == 'patch':
                    if ix is not None:
                        coord3d[2] = np.sqrt(
                            np.sum((self.vxe + self.vz0)[..., [0, 2]] ** 2, axis=-1)
                        ) - np.sqrt(np.sum(self.vz0[[0, 2]] ** 2, axis=-1))
                        coord3d[0] = 0
                    elif iy is not None:
                        coord3d[2] = np.sqrt(
                            np.sum((self.vye + self.vz0)[..., [1, 2]] ** 2, axis=-1)
                        ) - np.sqrt(np.sum(self.vz0[[1, 2]] ** 2, axis=-1))
                        coord3d[1] = 0
                elif self.geo == 'spherical':
                    if iy is not None:
                        coord3d[1] = 0
                        coord3d[0] = np.sqrt(
                            np.sum((self.vye + self.vz0)[..., [0, 1]] ** 2, axis=-1)
                        ) - np.sqrt(np.sum(self.vz0[[0, 1]] ** 2, axis=-1))
                    elif ix is not None:
                        v = self.vxe + self.vz0
                        ra = np.sqrt(np.sum(v[..., [0, 1]] ** 2, axis=-1))
                        rc = np.sqrt(np.sum(v[..., [0, 1, 2]] ** 2, axis=-1))
                        pa = np.arctan2(v[..., 1], v[..., 0])
                        y = ra * pa
                        pc = y / rc
                        sc = np.sin(pc)
                        cc = np.cos(pc)
                        coord3d[0] = 0
                        coord3d[1] = rc * sc
                        coord3d[2] = rc * cc - np.sqrt(
                            np.sum(self.vz0[[0, 1, 2]] ** 2, axis=-1)
                        )
                elif self.geo == 'cylindrical':
                    if iy is not None:
                        coord3d[1] = 0
                        coord3d[0] = np.sqrt(
                            np.sum((self.vye + self.vz0)[..., [0, 1]] ** 2, axis=-1)
                        ) - np.sqrt(np.sum(self.vz0[[0, 1]] ** 2, axis=-1))
                    elif ix is not None:
                        v = self.vxe + self.vz0
                        ra = np.sqrt(np.sum(v[..., [0, 1]] ** 2, axis=-1))
                        pa = np.arctan2(v[..., 1], v[..., 0])
                        y = ra * pa
                        coord3d[0] = 0
                        coord3d[1] = y

        none = slice(None)
        cut = list()
        coords = list()
        labels = list()
        types = list()
        if mode == 'voxel':
            voxels = np.full(self.shape, False)
        for i, j in enumerate((ix, iy, iz)):
            if j is None:
                cut.append(none)
                coords.append(coord3d[i])
                labels.append(label3d[i])
            else:
                if mode in ('shell', 'voxel',):
                    labels.append(label3d[i])
                    coords.append(coord3d[i])
                cut.append(j)
        cut = tuple(cut)
        if mode in ('voxel',):
            voxels[cut] = True
            cut = (none,) * 3
            x, y, z = coords
            xlabel, ylabel, zlabel = labels
        elif mode in ('shell',):
            x, y, z = coords
            z = z[cut]
            xlabel, ylabel, zlabel = labels
        elif mode in ('coord', 'project',):
            x, y = coords
            xlabel, ylabel = labels
        else:
            raise
        Q = Q[cut]
        x = x[cut]
        y = y[cut]
        if scale is None:
            scale = defaultscale
        if vmin is None:
            vmin = Q.min()
        if vmax is None:
            vmax = Q.max()
        if scale == 'log':
            Q = np.log10(Q)
            norm = Normalize
            label = f'log( {label} )'
            vmin, vmax = np.log10([vmin, vmax])
        elif scale == 'sci':
            norm = LogNorm
        else:
            norm = Normalize
        if unit is not None:
            label = f'{label} ({unit})'
        norm = norm(vmin=vmin, vmax=vmax)
        # if mirror == True:
        #     x = np.r_[-x[:0:-1], x]
        #     y = np.r_[ y[:0:-1], y]
        #     Q = np.r_[ Q[::-1,:], Q]
        # elif mirror == 4:
        #     x = np.r_[-x[:0:-1]-x[-1], -x[:0:-1], x, x[1:] + x[-1]]
        #     y = np.r_[ z[:-1], y[:0:-1], y, y[:0:-1]]
        #     Q = np.r_[ Q, Q[::-1,:], Q, Q[::-1,:]]

        if np.max(np.abs(x)) > 1e5:
            lunit = 'km'
            f = 1e-5
        elif np.max(np.abs(x)) > 1e3:
            lunit = 'm'
            f = 0.01
        else:
            f = 1
            lunit = 'cm'
        if f != 1:
            x = x * f
            y = y * f
            if xlim is not None:
                xlim = np.array(xlim) * f
            if ylim is not None:
                ylim = np.array(ylim) * f
            if mode in ('shell', 'voxel',):
                z = z * f
                if zlim is not None:
                    zlim = np.array(zlim) * f

        # set up figure
        if fig is None:
            fig = plt.figure()
        if mode in ('shell', 'voxel',):
            ax = fig.add_subplot(
                111, projection='3d', autoscale_on=False, box_aspect=(1, 1, 1),
            )
        else:
            ax = fig.add_subplot(111)
            # ax.set_aspect('equal')

        # make plots
        if mode in ('shell', 'voxel',):
            if mode == 'shell':
                cm = ax.plot_surface(
                    x,
                    y,
                    z,
                    facecolors=cmap(norm(Q)),
                    linewidth=0,
                    antialiased=False,
                    cmap=cmap,
                    shade=True,
                )
            elif mode == 'voxel':
                cm = ax.voxels(
                    x,
                    y,
                    z,
                    voxels,
                    facecolors=cmap(norm(Q)),
                    linewidth=0,
                    antialiased=False,
                    edgecolor='none',
                    cmap=cmap,
                    shade=True,
                )
                cm = next(iter(cm.values()))
            cm.norm = norm

            cb = fig.colorbar(cm, label=label, orientation='horizontal')
            ax.set_zlabel(f'{zlabel} ({lunit})')

            if xlim is None:
                xlim = [np.min(x), np.max(x)]
            if ylim is None:
                ylim = [np.min(y), np.max(y)]
            if zlim is None:
                zlim = [np.min(z), np.max(z)]

            xr = xlim[-1] - xlim[0]
            yr = ylim[-1] - ylim[0]
            zr = zlim[-1] - zlim[0]

            xm = 0.5 * (xlim[-1] + xlim[0])
            ym = 0.5 * (ylim[-1] + ylim[0])
            zm = 0.5 * (zlim[-1] + zlim[0])

            f = np.max((xr, yr, zr))

            xlim = (np.array(xlim) - xm) * f / (xr + TINY) + xm
            ylim = (np.array(ylim) - ym) * f / (yr + TINY) + ym
            zlim = (np.array(zlim) - zm) * f / (zr + TINY) + zm

            if zlim is not None:
                ax.set_zlim(zlim)
        else:
            cm = ax.pcolormesh(x.T, y.T, Q.T, norm=norm, cmap=cmap)
            cb = fig.colorbar(cm, label=label, orientation='horizontal')
        ax.set_xlabel(f'{xlabel} ({lunit})')
        ax.set_ylabel(f'{ylabel} ({lunit})')
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

        # -------- Added by Adelle to save T---------
        print(f'X = {x.T}, y = {y.T}, Q = {Q.T}')
        run = self.run_name
        qb = 1
        Thot = self.thot
        with open(
            f"data/data_solutiontemp_run{run}_qb{qb}_hs{Thot}.txt", "wb"
        ) as f:
            np.savetxt(
                f,
                Q.T,
                fmt="%.18e",
                delimiter=" ",
                header="T[0,:], .. T[x,:]",
                comments="# everything is in cgs\n",
            )

        fig.text(0.01, 0.99, self.geo, ha='left', va='top')
        fig.text(0.99, 0.99, mode, ha='right', va='top')
        fig.savefig(f'{run}_Temp{Thot}.pdf')
        if t is None:
            t = getattr(self, 't', None)
        if t is not None:
            fig.text(0.99, 0.99, time2human(t), ha='right', va='top')
        fig.tight_layout()
        if pos is not None:
            ax.set_position(pos)
        if cbpos is not None:
            cb.ax.set_position(cbpos)

    def plotly(self, q='T', T=None, net=None, scale=None, image=True):
        import plotly.graph_objects as go

        x = self.xc
        y = self.yc
        z = self.zc
        Q, label, unit, defaultscale, cmap = self.Q(q, T, net)

        if self.ny == 1:
            s = (1, self.nx, 1)
            x = np.tile(x, s)
            z = np.tile(z, s)
            Q = np.tile(Q, s)
            y = (self.yf[:, -1:, :] - self.yf[:, :1, :]) * (
                (np.arange(self.nx) + 0.5) / self.nx
            ).reshape((1, -1, 1))
            y = 0.25 * (y[:-1, :, :-1] + y[1:, :, :-1] + y[-1, :, 1:] + y[1:, :, 1:])

        vmin = np.min(Q)
        vmax = np.max(Q)

        if scale is None:
            scale = defaultscale
        if vmin is None:
            vmin = Q.min()
        if vmax is None:
            vmax = Q.max()
        if scale == 'log':
            Q = np.log10(Q + TINY)
            norm = Normalize
            label = f'log( {label} )'
            vmin, vmax = np.log10([vmin + TINY, vmax + TINY])
        if unit is not None:
            label = f'{label} ({unit})'

        if self.geo in ('cartesian', 'patch',):
            pass

        elif self.geo in ('cylindrical', 'spherical',):
            p = y / x
            d = x

            x = d * np.cos(p)
            y = d * np.sin(p)

            # x = self.xc.copy()
            # y = p * self.lx / (2*np.pi)

            # broken

        fig = go.Figure(
            data=go.Volume(
                x=x.flatten(),
                y=y.flatten(),
                z=z.flatten(),
                value=Q.flatten(),
                isomin=vmin,
                isomax=vmax,
                opacity=0.2,
                surface_count=21,
                caps=dict(
                    x_show=True, y_show=True, z_show=True, x_fill=1,
                ),  # with caps (default mode)
                colorbar=dict(title=label,),
                #         contour = dict(show=True), # slow
            )
        )
        fig.update_layout(
            scene_camera=dict(
                up=dict(x=0, y=0, z=500),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1, y=-2.5, z=1),
            ),
        )
        fig.update_layout(
            scene=dict(
                xaxis_title='width (cm)',
                yaxis_title='depth (cm)',
                zaxis_title='heigt (cm)',
            ),
        )
        fig.update_layout(scene_aspectmode='data')
        fig.show()
        self.fig = fig

        if image:
            from PIL import Image
            from io import BytesIO

            i = fig.to_image()
            i = Image.open(BytesIO(i))
            i.show()

    def plot2(
        self,
        q=('eps', 'T',),
        /,
        scale=(None, None),
        vmin=(None, None),
        vmax=(None, None),
        t=None,
        fig=None,
        T=None,
        net=None,
        pad=(0.1, 0.25),
        xlim=None,
        ylim=None,
    ):
        raise Exception('This routine has not yet been adjusted to 3D data.')
        if fig is None:
            fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        Qx = list()
        assert len(q) == 2
        for i, qx in enumerate(q):
            Q, label, unit, defaultscale, cmap = self.Q(qx, T, net)
            scalex = scale[i]
            vminx = vmin[i]
            vmaxx = vmax[i]
            if scalex is None:
                scalex = defaultscale
            if vminx is None:
                vminx = Q.min()
            if vmaxx is None:
                vmaxx = Q.max()
            if scalex == 'log':
                Q = np.log10(np.maximum(Q, TINY))
                norm = Normalize
                label = f'log( {label} )'
                vminx, vmaxx = np.log10([vminx, vmaxx])
            elif scalex == 'sci':
                norm = LogNorm
            else:
                norm = Normalize
            if unit is not None:
                label = f'{label} ({unit})'
            norm = norm(vmin=vminx, vmax=vmaxx)
            Qx.append(dict(Q=Q, label=label, norm=norm))

            x = self.xf
            z = self.zf
            if i == 0:
                x = -x[::-1]
                z = z[::-1]
                Q = Q[::-1, :]
            if self.geo in ('spherical', 'equatorial',):
                r = self.r + z
                p = x / r
                s = np.sin(p)
                c = np.cos(p)
                x = r * s
                z = r * c - self.r

            if np.max(np.abs(x)) > 1e5:
                lunit = 'km'
                x = x * 0.00001
                z = z * 0.00001
            elif np.max(np.abs(x)) > 1e3:
                lunit = 'm'
                x = x * 0.01
                z = z * 0.01
            else:
                lunit = 'cm'

            cm = ax.pcolormesh(x.T, z.T, Q.T, norm=norm, cmap=cmap)
            cb = fig.colorbar(cm, label=label, orientation='horizontal', pad=pad[i])
        ax.set_xlabel(f'Horizontal Distance ({lunit})')
        ax.set_ylabel(f'Height ({lunit})')
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        fig.text(0.01, 0.99, self.geo, ha='left', va='top')
        if t is None:
            t = getattr(self, 't', None)
        if t is not None:
            fig.text(0.99, 0.99, time2human(t), ha='right', va='top')
        fig.tight_layout()

    @staticmethod
    def examples():

        # from heat import Diffuse as D

        # single pole
        r = 200
        d = D(
            geo='s',
            nz=300,
            nx=round(np.pi * 200 * 2),
            method='solve4',
            sparse=True,
            r=r,
            lx=np.pi * r,
            xhot=(r + 150) * 0.5,
            mbase='F',
            thot=1e8,
        )
        d.plot()

        # two poles
        r = 200
        d = D(
            geo='s',
            nz=300,
            nx=round(np.pi * 200),
            method='solve4',
            sparse=True,
            r=r,
            lx=0.5 * np.pi * r,
            xhot=(r + 150) * 0.5,
            mbase='F',
            thot=1e8,
        )
        d.plot(mirror=4)

        # equator
        r = 200
        d = D(
            geo='e',
            nz=300,
            nx=round(np.pi * 200),
            method='solve4',
            sparse=True,
            r=r,
            lx=0.5 * np.pi * r,
            xhot=(r + 150) * 0.5,
            mbase='F',
            thot=1e8,
        )
        d.plot(mirror=4)

        # sphere
        r = 1
        d = D(
            geo='s',
            nz=300,
            nx=round(np.pi * 200 * 2),
            method='solve4',
            sparse=True,
            r=r,
            lx=r * np.pi,
            xhot=(r + 150) * 0.1,
            thot=1e8,
            mbase='F',
            fbase=0,
        )
        d.plot()

        # bend
        r = 1000
        rmin = 200
        res = 0.5 * 10
        h = 150
        w = np.pi * rmin
        xhot = 20
        d = D(
            geo='s',
            nz=round(h / res),
            nx=round(w / res),
            method='solve4',
            r=r,
            lx=w,
            lz=h,
            xhot=xhot * (r + h) / r,
            thot=1e8,
            mbase='F',
        )
        d.plot(vmax=1e8, vmin=3e6)

        # T evolution movie
        e = D(nx=80, nz=30, xhot=0, thot=1e8)
        e.xhot = 20
        e.evo(
            movie='T1e8x20.webm',
            tstop=86400,
            pkwargs=dict(vmin=4e6, vmax=1e8),
            fkwargs=dict(size=(800, 320)),
            store=True,
        )

        # burn movie
        e = D(geo='c', nx=80, nz=30, xhot=0, thot=2e9, method='solve4', net='fancy')
        e.xhot = 100
        e.evo(dTr=1, dt0=1e-8, tstop=86400)
        e.store2movie(
            #'T2e9x100cX.mp4',
            'y.mp4',
            mkwargs=dict(delay=1 / 60),
            fkwargs=dict(size=(800, 400)),
            pkwargs=dict(q=('c12', 'mg24'), pad=(0.15, 0.2), vmin=(0, 0), vmax=(1, 1)),
            sel=slice(None, None, 1),
        )
        e.store2movie(
            'T2e9x100ceT.mp4',
            mkwargs=dict(delay=1 / 60),
            fkwargs=dict(size=(800, 400)),
            pkwargs=dict(vmin=(7e6, 1e10), pad=(0.15, 0.2), vmax=(2e9, 5e17)),
            sel=slice(None, None, 1),
        )


def bend_movie(filename='bend.webm', fps=60, res=5, steps=600, mp=False, **kwargs):
    rmax = 1e6
    rmin = 200
    rr = rmax / (np.arange(steps) / (steps - 1) * (rmax / rmin - 1) + 1)
    h = 150
    w = np.pi * rmin

    fkwargs = dict(xhot=20, thot=1e8, mbase='F',)
    fkwargs.update(kwargs)

    def generator(r, fig):
        d = Diffuse(
            geo='s',
            nz=round(h / res),
            nx=round(w * (h + r) / (r * res)),
            method='solve4',
            r=r,
            lx=w,
            lz=h,
            **fkwargs,
        )
        d.plot(
            vmax=1e8,
            vmin=3e6,
            xlim=[-w * 1.1, w * 1.1],
            ylim=[-(2 * rmin + h) * 1.1, h * 1.1],
            fig=fig,
            pos=[0.1, 0.3, 0.875, 0.65],
            cbpos=[0.1, 0.1, 0.875, 0.1],
        )

    make_movie(
        filename,
        generator=generator,
        canvas=MPLCanvas,
        data='r',
        values=rr,
        fps=fps,
        mp=mp,
    )


def hole_movie(
    filename='hole.webm',
    geo='s',
    fps=60,
    res=5,
    steps=600,
    xhotf=0.1,
    mp=False,
    **kwargs,
):
    rmax = 1e6
    rmin = 0.1
    rr = np.exp(np.arange(steps) / (steps - 1) * (np.log(rmin) - np.log(rmax))) * rmax
    h = 150

    fkwargs = dict(thot=1e8, mbase='F', geo=geo,)
    fkwargs.update(kwargs)

    pkwargs = dict()
    if geo == 'e':
        pkwargs['mirror'] = 4
        angle = 0.5 * np.pi
    elif geo == 's':
        mirror = 2
        angle = np.pi
    else:
        raise AttributeError(f'Geometry not defined: {geo}')

    def generator(r, fig):
        w = r * angle
        x = r + h
        d = Diffuse(
            nz=round(h / res),
            nx=round(w * 200 / (r * res)),
            method='solve4',
            r=r,
            lx=w,
            lz=h,
            xhot=x * xhotf,
            **fkwargs,
        )
        d.plot(
            vmax=1e8,
            vmin=3e6,
            xlim=1.05 * x * np.array([-1, 1]),
            ylim=1.05 * x * np.array([-1, 1]) - r,
            fig=fig,
            pos=[0.1, 0.2, 0.875, 0.75],
            cbpos=[0.1, 0.025, 0.875, 0.1],
            **pkwargs,
        )

    make_movie(
        filename,
        generator=generator,
        canvas=MPLCanvas,
        ckwargs=dict(size=(600, 720)),
        data='r',
        values=rr,
        fps=fps,
        mp=mp,
    )
