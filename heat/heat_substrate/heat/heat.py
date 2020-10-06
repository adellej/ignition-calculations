

import time # always good to have
import resource
import copy
import pickle
from pathlib import Path

from functools import partial

import numpy as np
import numpy.linalg

from scipy.optimize import root
from scipy.linalg import solve
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse import diags
from matplotlib import pylab as plt
from matplotlib.colors import LogNorm, Normalize

from human import time2human
import color

from physconst import KB, RK
from movie import MovieWriter
from movie import make_movie
from movie import MPLCanvas

from .net import Net3aFFNs, Net3aFFNt, Net3aC12

from .numeric import *

class _Diffuse_Result(): pass

class Diffuse(object):
    def __init__(
            self,
            nx = 80,
            nz = 30,
            lx = 400,
            lz = 150,
            geo = 'cart',
            tbase = 1e8,
            fbase = 1e19,
            mbase = 'F',
            ttop = 1e6,
            thot = 5e6,
            xhot = 20,
            tol = 1e-6,
            op = None,
            rhoflag = 'fit',
            r = 1e6,
            method = 'solve4',
            normalize = False,
            T = None,
            sparse = True,
            initial = False,
            jac = True,
            xres = 'constant',
            zres = 'constant',
            root_options = None,
            root4 = True,
            Y = 1,
            net = 'static',
            ):

        if initial is None:
            return

        # we use 'f' for face and 'c' for centre

        nx1 = nx + 1
        nz1 = nz + 1

        self.shape = (nx, nz)

        self.nx = nx
        self.nz = nz
        self.nx1 = nx1
        self.nz1 = nz1

        self.lx = lx
        self.lz = lz

        self.Y = Y

        if net == 'static':
            net = Net3aFFNs
            abu = (Y / 4, )
        elif net == 'dynamic':
            net = Net3aFFNt
            abu = (Y / 4, (1 - Y) / 12)
        elif net == 'fancy':
            net = Net3aC12
            abu = (Y / 4, (1 - Y) / 12, 0)
        else:
            raise AttributeError(f'Unknown network type: {net}')
        self.net = net(abu, (nx, nz))

        # basic 1D linear coordinate grid through (0,0), naming: ends in "1"

        if xres == 'progressive':
            nnx = (nx * (nx + 1)) // 2
            dx1 = lx * (np.arange(nx) + 1) / nnx
        elif xres == 'constant':
            dx1 = np.tile(lx / nx, nx)
        else:
            raise AttributeError(f'Unknown xres = {xres}')
        xf1 = np.insert(np.cumsum(dx1), 0, 0)
        xc1 = 0.5 * (xf1[1:] + xf1[:-1])

        if zres == 'progressive':
            nnz = (nz * (nz + 1)) // 2
            dz1 = lz * (np.arange(nz)[::-1] + 1) / nnz
        elif zres == 'constant':
            dz1 = np.tile(lz / nz, nz)
        else:
            raise AttributeError(f'Unknown zres = {zres}')
        zf1 = np.insert(np.cumsum(dz1), 0, 0)
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

        self.rhoflag = rhoflag
        self.kb = 1.231e15 * (2 ** (4 / 3))  # ideal gas constant for pure helium
        self.g = 2.426e14  # surface gravity, cgs
        self.kappa = 0.136  # cm^2/g, assume constant opacity
        self.ARAD = 7.566e-15  # radiation constant
        self.CLIGHT = 2.998e10  # speed of light

        self.cst4 = - self.ARAD * self.CLIGHT / ( 3 * self.kappa )
        self.cst  =   0.5 * self.cst4

        rhof1 = self.rho_ini(zf1)
        rhoc1 = self.rho_ini(zc1)
        rhofi1 = 1 / rhof1
        rhoci1 = 1 / rhoc1

        rhoc  = np.tile(rhoc1,  (nx , 1))
        rhoci = np.tile(rhoci1, (nx , 1))

        rhoz  = np.tile(rhof1,  (nx , 1))
        rhozi = np.tile(rhofi1, (nx , 1))

        rhox  = np.tile(rhoc1,  (nx1, 1))
        rhoxi = np.tile(rhoci1, (nx1, 1))

        # geometric factors:
        #   az, az - areas on sides
        #   v - volumes
        dz = np.tile(dz1, (nx1, 1))
        zf = np.tile(zf1, (nx1, 1))
        zc = np.tile(zc1, (nx , 1))

        ax = np.ndarray((nx1, nz ))
        az = np.ndarray((nx , nz1))
        v  = np.ndarray((nx , nz ))
        if geo in ('d', 'cart', 'cartesian', ):
            # 'd' for 'Decartes'
            geo = 'cartesian'
            dx = np.tile(dx1, (nz1, 1)).T
            xf = np.tile(xf1, (nz1, 1)).T
            xc = np.tile(xc1, (nz , 1)).T
            dl = lx
            ax[:, :] = dz * dl
            az[:, :] = dx * dl
            v[:, :]  = np.outer(dx1 * dl, dz1)
        elif geo in ('c', 'cyl', 'cylindrical', ):
            geo = 'cylindrical'
            dx = np.tile(dx1, (nz1, 1)).T
            xf = np.tile(xf1, (nz1, 1)).T
            xc = np.tile(xc1, (nz,  1)).T
            fz1 = xf1**2 * np.pi
            dfz1 = (fz1[1:] - fz1[:-1])
            az[:, :] = dfz1[:, np.newaxis]
            ax[:, :] = 2 * np.pi * np.outer(xf1, dz1)
            v[:, :] = np.outer(dfz1, dz1)
        elif geo in ('s', 'sph', 'spherical', ):
            geo = 'spherical'
            assert lx <= r * np.pi
            dx = dx1[:, np.newaxis] * ((zf1 + r) / (zf1[0] + r))[np.newaxis, :]
            xc = xc1[:, np.newaxis] * ((zc1 + r) / (zf1[0] + r))[np.newaxis, :]
            xf = xf1[:, np.newaxis] * ((zf1 + r) / (zf1[0] + r))[np.newaxis, :]
            h = (1 - np.cos(xf1 / r))[:, np.newaxis] * (r + zf1)[np.newaxis, :]
            fz = h * (r + zf1)[np.newaxis, :] * 2 * np.pi
            az[:, :] = fz[1:] - fz[:-1]
            w = np.sin(xf1 / r)[:, np.newaxis] * (r + zf1)[np.newaxis, :]
            fx = w * (r + zf1)[np.newaxis, :] * np.pi
            ax[:, :] = fx[:, 1:] - fx[:, :-1]
            vt = h * (r + zf1)[np.newaxis, :]**2 * (2 * np.pi / 3)
            dvx = vt[1:] - vt[:-1]
            v[:, :] = dvx[:, 1:] - dvx[:, :-1]
        elif geo in ('e', 'equ', 'equatorial', ):
            geo = 'equatorial'
            assert lx <= 0.5 * r * np.pi
            dx = dx1[:, np.newaxis] * ((zf1 + r) / (zf1[0] + r))[np.newaxis, :]
            xc = xc1[:, np.newaxis] * ((zc1 + r) / (zf1[0] + r))[np.newaxis, :]
            xf = xf1[:, np.newaxis] * ((zf1 + r) / (zf1[0] + r))[np.newaxis, :]
            h = np.sin(xf1 / r)[:, np.newaxis] * (r + zf1)[np.newaxis, :]
            fz = h * (r + zf1)[np.newaxis, :] * 2 * np.pi
            az[:, :] = fz[1:] - fz[:-1]
            w = np.cos(xf1 / r)[:, np.newaxis] * (r + zf1)[np.newaxis, :]
            fw = w * (r + zf1)[np.newaxis, :] * np.pi
            ax[:, :] = fw[:, 1:] - fw[:, :-1]
            vt = h * (r + zf1)[np.newaxis, :]**2 * (2 * np.pi / 3)
            dvx = vt[1:] - vt[:-1]
            v[:, :] = dvx[:, 1:] - dvx[:, :-1]
        else:
            raise AttributeError(f'Unknown Geometry "{geo}"')

        # why not use d(xc, zc)?
        dxf = np.ndarray((nx + 1, nz))
        dxf[1:-1, :] = 0.25 * (dx[ 1:, 1:] + dx[:-1, 1:] + dx[ 1:,:-1] + dx[:-1,:-1])
        dxf[   0, :] = 0.5 *  (dx[  0,:-1] + dx[  0, 1:])
        dxf[  -1, :] = 0.5 *  (dx[ -1,:-1] + dx[ -1, 1:])
        self.dxfi = 1 / dxf

        dzf = np.ndarray((nx, nz + 1))
        dzf[:, 1:-1] = 0.25 * (dz[ 1:, 1:] + dz[:-1, 1:] + dz[ 1:,:-1] + dz[:-1,:-1])
        dzf[:,    0] = 0.5  * (dz[:-1,  0] + dz[ 1:,  0])
        dzf[:,   -1] = 0.5  * (dz[:-1, -1] + dz[ 1:, -1])
        self.dzfi = 1 / dzf

        m = v * rhoc

        f0 = az[0,0]
        v0 = v[0,0]
        if normalize:
            #   f0 - area to which units are normalised
            #   v0 - v[0,0]
            f0i = 1 / f0
            #v0i = 1 / v0
            ax *= f0i
            az *= f0i
            #v *= f0i
            m *= f0i

        self.geo = geo
        self.ax = ax
        self.az = az
        #self.v = v
        #self.f0 = f0
        #self.v0 = v0
        #self.vl = v0 / f0
        self.m = m
        self.rhoc = rhoc
        self.rhoxi = rhoxi
        self.rhozi = rhozi
        self.xc = xc
        self.xf = xf
        self.zf = zf
        self.xcsurf = 0.5 * (xf[1:, -1] + xf[:-1, -1])

        self.Cv = m * 1.5 * RK * (3/4)

        if T is None or T.shape != v.shape:
            # initialise T (crudely)
            dyy = rhoz * dzf
            yy = np.cumsum(dyy[:,::-1], axis=1)[:, ::-1]
            yy = np.hstack((yy, np.zeros((nx, 1))))
            yyc = yy[:,1:-1]

            ihot = np.where(self.xcsurf > xhot)[0][0]
            ttop = np.array([thot]*ihot + [ttop]*(nx - ihot))

            if mbase == 'F':
                T4 = 3 * self.kappa * fbase * yyc / (self.ARAD * self.CLIGHT) + quad(ttop)[:, np.newaxis]
            else:
                T4 = quad(tbase) * yyc / yy[:, 0][:, np.newaxis] + quad(ttop)[:, np.newaxis]
            T = qqrt(T4)

        self.method = method

        if initial:
            self.T = T
            return

        # solve
        starttime = time.time()
        if method in ('df-sane', 'lm', 'broyden1', 'broyden2', 'diagbroyden',
                      'linearmixing', 'excitingmixing', 'anderson', 'hybr' ):
            if method in (
                    'df-sane', 'broyden1', 'broyden2', 'diagbroyden',
                    'linearmixing', 'excitingmixing', 'anderson'):
                self.jac = None
            if method in ('lm', 'hybr',):
                if self.jac is True:
                    self.sparse = False

            t = np.ndarray((2,3))
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
            print(f'[DIFFUSE] [{method}] total:   time: {time2human(x[0]):>8s} real, {time2human(x[1]):>8s} user, {time2human(x[2]):>8s} sys, {time2human(x[3]):>8s} cpu total.')
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

        self.T = solution.x.reshape((nx, nz))

    def copy_plot_data(self):
        new = self.__class__(initial = None)
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
                rho_t * (1 + (self.g / (4 * self.kb * rho_t ** (1 / 3))) * (self.lz-z))
                )
        elif self.rhoflag == "fit":
            rho = rho0 * np.exp(beta * z ** alpha)
        else:
            rho = rhoflag
        return rho

    def func(self, T_, dt=None, T4=True):
        T = T_.reshape((self.nx, self.nz))

        i0 = slice(0, -1)
        i1 = slice(1, None)
        ii = slice(1, -1)
        ia = slice(None, None)

        icx  = (ii, ia)
        icz  = (ia, ii)
        icia = (ii, ia)
        icai = (ia, ii)
        ic0a = (i0, ia)
        ic1a = (i1, ia)
        ica0 = (ia, i0)
        ica1 = (ia, i1)

        if not T4:
            Tfx3 = cube(T[ic1a] + T[ic0a])
            Tfz3 = cube(T[ica1] + T[ica0])

        dTfx = T[ic1a] - T[ic0a]
        dTfz = T[ica1] - T[ica0]

        Fx = np.tile(np.nan, (self.nx1, self.nz ))
        Fz = np.tile(np.nan, (self.nx , self.nz1))

        if T4:
            cst = self.cst4
        else:
            cst = self.cst

        cfx = cst * self.rhoxi * self.ax * self.dxfi
        cfz = cst * self.rhozi * self.az * self.dzfi

        if not T4:
            dTfx *= Tfx3
            dTfz *= Tfz3

        Fx[icia] = cfx[icia] * dTfx
        Fz[icai] = cfz[icai] * dTfz

        if T4:
            eps, deps = self.net.epsdt4_T4(T, self.rhoc, dt)
        else:
            eps, deps = self.net.epsdt_T(T, self.rhoc, dt)

        S = eps * self.m

        # boundary conditions
        Fx[[0, -1], :] = 0

        if self.mbase == 'F':
            Fz[:, 0] = self.fbase * self.az[:, 0]
        else:
            if T4:
                tbase4 = np.tile(quad(self.tbase), self.nx)
                Fz[:, 0] = cfz[:, 0] * (T[:, 0] - tbase4)
            else:
                tbase = np.tile(self.tbase, self.nx)
                Fz[:, 0] = cfz[:, 0] * cube(T[:, 0] + tbase) * (T[:, 0] - tbase)

        ihot = np.where(self.xcsurf > self.xhot)[0][0]
        if T4:
            ttop4 = quad(np.array([self.thot]*ihot + [self.ttop]*(self.nx-ihot)))
            Fz[:, -1] = cfz[:, -1] * (ttop4 - T[:, -1])
        else:
            ttop = np.array([self.thot] * ihot + [self.ttop] * (self.nx-ihot))
            Fz[:, -1] = cfz[:, -1] * cube(ttop + T[:, -1]) * (ttop - T[:, -1])

        # sum all terms
        F = (S + Fx[ic0a] - Fx[ic1a] + Fz[ica0] - Fz[ica1]).flatten()

        if not self.jac:
            return F

        # ====================
        #       Jacobian
        # ====================
        cfx[[0,-1], :] = 0
        if self.mbase == 'F':
            cfz[:, 0] = 0

        if not hasattr(self, '_s'):
            diag = np.mgrid[0:self.nx, 0:self.nz]
            i00 = tuple(np.r_[diag, diag])
            imx = (diag[0][ic1a], diag[1][ic1a], diag[0][ic1a] - 1, diag[1][ic1a]    )
            ipx = (diag[0][ic0a], diag[1][ic0a], diag[0][ic0a] + 1, diag[1][ic0a]    )
            imz = (diag[0][ica1], diag[1][ica1], diag[0][ica1]    , diag[1][ica1] - 1)
            ipz = (diag[0][ica0], diag[1][ica0], diag[0][ica0]    , diag[1][ica0] + 1)

            nn = self.nx * self.nz
            if self.sparse:
                m = np.arange(nn).reshape(self.nx, self.nz)
                def s_(ii):
                    return (m[ii[:2]].flatten(), m[ii[2:]].flatten())
                i00 = s_(i00)
                imx = s_(imx)
                ipx = s_(ipx)
                imz = s_(imz)
                ipz = s_(ipz)
                s = -1
            else:
                s= None
            class _Store(object): pass
            _s = _Store()
            _s.s   = s
            _s.i00 = i00
            _s.ipx = ipx
            _s.imx = imx
            _s.ipz = ipz
            _s.imz = imz
            _s.nn  = nn
            self._s = _s
        else:
            _s = self._s
            s   = _s.s
            nn  = _s.nn
            i00 = _s.i00
            ipx = _s.ipx
            imx = _s.imx
            ipz = _s.ipz
            imz = _s.imz

        if self.sparse:
            dF = lil_matrix((nn, nn))
        else:
            dF = np.zeros((self.nx, self.nz, self.nx, self.nz))

        if T4:
            dF[i00] = (deps * self.m + cfx[ic0a] + cfx[ic1a] + cfz[ica0] + cfz[ica1]).reshape(s)
            dF[imx] = - cfx[icx].reshape(s)
            dF[ipx] = - cfx[icx].reshape(s)
            dF[imz] = - cfz[icz].reshape(s)
            dF[ipz] = - cfz[icz].reshape(s)
        else:
            T2  = T**2
            T3  = T2 * T
            T32 = T3 * 2
            T34 = T3 * 4
            T26 = T2 * 6

            Tmx   = np.vstack((np.zeros((1, self.nz)), T  [ic0a]))
            Tmx32 = np.vstack((np.zeros((1, self.nz)), T32[ic0a]))
            Tpx   = np.vstack((T  [ic1a], np.zeros((1, self.nz))))
            Tpx32 = np.vstack((T32[ic1a], np.zeros((1, self.nz))))
            if self.mbase == 'F':
                Tmz   = np.hstack((np.zeros((self.nx, 1)), T  [ica0]))
                Tmz32 = np.hstack((np.zeros((self.nx, 1)), T32[ica0]))
            else:
                Tmz   = np.hstack((           tbase.reshape(self.nx, 1), T  [ica1], ))
                Tmz32 = np.hstack(((2 * cube(ttop)).reshape(self.nx, 1), T32[ica1], ))
            Tpz   = np.hstack((T  [ica1],             ttop.reshape(self.nx, 1)))
            Tpz32 = np.hstack((T32[ica1], (2 * cube(ttop)).reshape(self.nx, 1)))

            dF[i00] = (
                deps * self.m
                + cfx[ic0a] * (T34 + T26 * Tmx - Tmx32)
                + cfx[ic1a] * (T34 + T26 * Tpx - Tpx32)
                + cfz[ica0] * (T34 + T26 * Tmz - Tmz32)
                + cfz[ica1] * (T34 + T26 * Tpz - Tpz32)
                ).reshape(s)
            dF[imx] = (cfx[icx] * (T32[ic1a] - T[ic1a] * T26[ic0a] - T34[ic0a])).reshape(s)
            dF[ipx] = (cfx[icx] * (T32[ic0a] - T[ic0a] * T26[ic1a] - T34[ic1a])).reshape(s)
            dF[imz] = (cfz[icz] * (T32[ica1] - T[ica1] * T26[ica0] - T34[ica0])).reshape(s)
            dF[ipz] = (cfz[icz] * (T32[ica0] - T[ica0] * T26[ica1] - T34[ica1])).reshape(s)

        if self.sparse:
            dF = dF.tocsc()
        else:
            dF = dF.reshape((nn, nn))
        return F, dF

    def solve(self, T, op=None, tol=1e-6, f = None):
        if f is None:
            f = self.fun4
        if self.sparse:
            solver = partial(spsolve, use_umfpack=False)
        else:
            solver = solve
        t = np.ndarray((4,3))
        dt = np.zeros((3,3))
        for i in range(1000):
            self.iter = i
            use = resource.getrusage(resource.RUSAGE_SELF)
            t[0,:] = (time.time(), use.ru_utime, use.ru_stime)
            b, A = f(T)
            use = resource.getrusage(resource.RUSAGE_SELF)
            t[1,:] = (time.time(), use.ru_utime, use.ru_stime)
            c = solver(A, b).reshape((self.nx, self.nz))
            use = resource.getrusage(resource.RUSAGE_SELF)
            t[2,:] = (time.time(), use.ru_utime, use.ru_stime)
            ierr = np.unravel_index(np.argmax(np.abs(c / T)), (self.nx, self.nz))
            err = c[ierr] / T[ierr]

            # if self.iter == 1:
            #     result = _Diffuse_Result()
            #     result.x = np.zeros(self.shape)
            #     result.message='debug'
            #     result.success=False
            #     self.debug = result
            #     result.A = A
            #     result.b = b
            #     result.c = c
            #     return result

            if op is not None:
                opx = np.minimum(1, op / np.abs(err))
                T = T - c * opx
            else:
                opx = 1
                T -= c
            use = resource.getrusage(resource.RUSAGE_SELF)
            t[3,:] = (time.time(), use.ru_utime, use.ru_stime)
            dt[:,:] += t[1:,:] - t[:-1,:]
            print(f'[DIFFUSE.SOLVE] {i:03d}: err = {err:+8e}, T = {T[ierr]:+8e}, opx = {opx:+8e}, ierr = {ierr}')
            if np.isnan(err):
                raise Exception('No convergence')
            if np.abs(err) < tol:
                break
        else:
            raise Exception('No convergence')
        dt = np.hstack((dt, np.sum(dt[:,1:], axis=1)[:, np.newaxis]))
        dt = np.vstack((dt, np.sum(dt[:,:], axis=0)[np.newaxis, :]))
        for m,x in zip(['setup','solve', 'update', 'total'], dt):
            print(f'[DIFFUSE.SOLVE] {m:<6s}  time: {time2human(x[0]):>8s} real, {time2human(x[1]):>8s} user, {time2human(x[2]):>8s} sys, {time2human(x[3]):>8s} cpu total.')
        result = _Diffuse_Result()
        result.x = T
        result.nit = i+1
        result.message = 'converged'
        result.success=True
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

    def evo_step(self, T, dT=None, dt=1e-4, t=None, maxit=20):
        """
        Henyey-style solver (implicit)
        """
        T0 = T.flatten()
        T0i = 1 / T0

        dti = 1 / dt
        Cv0 = self.Cv.flatten() * dti
        Cv1 = diags(Cv0, format = 'csc')

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
                A = dF - diags(Cv04 * Tnq / Tn, format = 'csc')
            else:
                b = F - Cv0 * dT
                A = dF - Cv1
            c = solver(A, b)
            Tn = Tn - c
            dT = Tn - T0
            dTs = np.max(np.abs(c ) * T0i)
            dTr = np.max(np.abs(dT) * T0i)
            print(f'[EVO_STEP] [{i:04}] dTr = {dTr:+8e}, dTs = {dTs:+8e}')
            if dTs < 1e-6:
                break
        else:
            return None, 0, 0
        Tn = Tn.reshape((self.nx, self.nz))
        return Tn, dT, dTr

    def evo(self,
            tstop = 1000,
            nstep = None,
            dt0 = None,
            dTr = None,
            movie = None,
            mkwargs = dict(),
            pkwargs = dict(),
            fkwargs = dict(),
            dtmax = None,
            store = True,
            t = None,
            dtf = 0.5 * (np.sqrt(5) + 1),
            ):
        if t is None:
            t = getattr(self, 't', 0)
        self.t = t
        if dt0 is None:
            dt0 = getattr(self, 'dt', 1e-4)
        dt = dt0
        dT = getattr(self, 'dT', None)
        xt = np.ndarray((5,3))
        use = resource.getrusage(resource.RUSAGE_SELF)
        xt[:,:] = np.array((time.time(), use.ru_utime, use.ru_stime))[np.newaxis, :]
        if movie is not None:
            fkw = dict(size=(800,320))
            fkw = dict(fkw, **fkwargs)
            mkw = dict(delay = 0.1)
            mkw = dict(mkw, **mkwargs)
            pkw = dict(q='T', scale='log')
            pkw = dict(pkw, **pkwargs)
            q = pkw.pop('q', 'T')
            movie = MovieWriter(
                movie,
                canvas = MPLCanvas(**fkw),
                generator = self.plot,
                gargs = (q,),
                gkwargs = pkw,
                **mkw,
                )
            movie.write(gkwargs = dict(t=t))
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
            nstop = 2**30
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
            self.eps = self.net.epsdt(self.T, self.rhoc, dt, update = True)
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
            print(f'[EVO] [{i:06d}] time = {t:+8e}, dtn = {dt:+8e}, real: {time2human(x[0]):>8s}, CPU: {time2human(x[3]):>8s}')
            if movie is not None:
                movie.write(gkwargs = dict(t=t))
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
            print(f'[EVO] movie:   time: {time2human(x[0]):>8s} real, {time2human(x[1]):>8s} user, {time2human(x[2]):>8s} sys, {time2human(x[3]):>8s} cpu total.')
        use = resource.getrusage(resource.RUSAGE_SELF)
        xt[-1,:] = (time.time(), use.ru_utime, use.ru_stime)
        x = xt[-1] - xt[0]
        x = np.hstack((x, np.sum(x[1:])))
        print(f'[EVO] total:   time: {time2human(x[0]):>8s} real, {time2human(x[1]):>8s} user, {time2human(x[2]):>8s} sys, {time2human(x[3]):>8s} cpu total.')

    def store2movie(
            self,
            movie=None,
            mkwargs = dict(),
            pkwargs = dict(),
            fkwargs = dict(),
            sel = slice(None),
            mp = True,
            ):
        xt = np.ndarray((2, 3))
        use = resource.getrusage(resource.RUSAGE_SELF)
        usc = resource.getrusage(resource.RUSAGE_CHILDREN)
        xt[:,:] = np.array((
                time.time(),
                use.ru_utime + usc.ru_utime,
                use.ru_stime + usc.ru_stime,
                ))[np.newaxis, :]
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
            generator = plotter,
            canvas = MPLCanvas,
            ckwargs = fkw,
            gargs = (q,),
            gkwargs = pkw,
            values = self.store[sel],
            data = 'kwargs',
            **mkw,
            )
        use = resource.getrusage(resource.RUSAGE_SELF)
        usc = resource.getrusage(resource.RUSAGE_CHILDREN)
        xt[1,:] = np.array((
                time.time(),
                use.ru_utime + usc.ru_utime,
                use.ru_stime + usc.ru_stime,
                ))[np.newaxis, :]
        xm = xt[1] - xt[0]
        x = np.hstack((xm, np.sum(xm[1:])))
        print(f'[EVO] movie:   time: {time2human(x[0]):>8s} real, {time2human(x[1]):>8s} user, {time2human(x[2]):>8s} sys, {time2human(x[3]):>8s} cpu total.')

    def Q(self, q, T=None, net=None):
        if T is None:
            T = self.T.copy()
        if net is None:
            net = self.net
        cmap_ = None
        cmap = None
        if q == 'eps':
            Q = net.eps(T, self.rhoc)
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
            Q = (Q / e * T)
            label = r'$\frac{\mathrm{d}\,\ln\,\varepsilon_{\mathrm{nuc}}}{\mathrm{d}\,\ln\,\,T}$'
            unit = None
            defaultscale = 'lin'
        elif q == 'dleps':
            e, Q = net.eps_T(T, self.rhoc)
            Q = (Q / e * T)
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
        else:
            raise AttributeError(f'Unknown plot quantity: {q}')
        if cmap_ is not None:
            cmap = cmap_
        else:
            if cmap is None:
                cmap = color.colormap('cividis')
            cmap = color.WaveFilter(cmap, nwaves = 40, amplitude=0.2)
        return Q, label, unit, defaultscale, cmap

    def plot(
            self,
            q='T',
            /,
            scale=None,
            mirror=True,
            fig=None,
            t=None,
            vmin=None,
            vmax=None,
            T=None,
            net=None,
            xlim=None,
            ylim=None,
            pos=None,
            cbpos=None,
            ):
        if fig is None:
            fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        Q, label, unit, defaultscale, cmap = self.Q(q, T, net)
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
        x = self.xf
        z = self.zf
        if mirror == True:
            x = np.r_[-x[:0:-1], x]
            z = np.r_[ z[:0:-1], z]
            Q = np.r_[ Q[::-1,:], Q]
        elif mirror == 4:
            x = np.r_[-x[:0:-1]-x[-1], -x[:0:-1], x, x[1:] + x[-1]]
            z = np.r_[ z[:-1], z[:0:-1], z, z[:0:-1]]
            Q = np.r_[ Q, Q[::-1,:], Q, Q[::-1,:]]

        if self.geo in ('spherical', 'equatorial', ):
            r = (self.r + z)
            p = x / r
            s = np.sin(p)
            c = np.cos(p)
            x = (r * s)
            z = (r * c - self.r)

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
            z = z * f
            if xlim is not None:
               xlim = np.array(xlim) * f
            if ylim is not None:
               ylim = np.array(ylim) * f

        cm = ax.pcolormesh(x.T, z.T, Q.T, norm=norm, cmap=cmap)
        cb = fig.colorbar(cm, label = label, orientation='horizontal')
        ax.set_xlabel(f'Horizontal Distance ({lunit})')
        ax.set_ylabel(f'Height ({lunit})')
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        fig.text(0.01,0.99,self.geo, ha='left', va='top')
        if t is None:
            t = getattr(self, 't', None)
        if t is not None:
            fig.text(0.99,0.99,time2human(t), ha='right', va='top')
        fig.tight_layout()
        if pos is not None:
            ax.set_position(pos)
        if cbpos is not None:
            cb.ax.set_position(cbpos)

    def plot2(
            self,
            q=('eps', 'T', ),
            /,
            scale=(None, None),
            vmin=(None,None),
            vmax=(None,None),
            t=None,
            fig=None,
            T=None,
            net=None,
            pad = (0.1, 0.25),
            xlim=None,
            ylim=None,
            ):
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
            Qx.append(
                dict(Q=Q, label=label, norm=norm)
                )

            x = self.xf
            z = self.zf
            if i == 0:
                x = -x[::-1]
                z =  z[::-1]
                Q =  Q[::-1,:]
            if self.geo in ('spherical', 'equatorial', ):
                r = (self.r + z)
                p = x / r
                s = np.sin(p)
                c = np.cos(p)
                x = (r * s)
                z = (r * c - self.r)

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
            cb = fig.colorbar(cm, label = label, orientation='horizontal', pad=pad[i])
        ax.set_xlabel(f'Horizontal Distance ({lunit})')
        ax.set_ylabel(f'Height ({lunit})')
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        fig.text(0.01,0.99,self.geo, ha='left', va='top')
        if t is None:
            t = getattr(self, 't', None)
        if t is not None:
            fig.text(0.99,0.99,time2human(t), ha='right', va='top')
        fig.tight_layout()


    @staticmethod
    def examples():

        # from heat import Diffuse as D

        # single pole
        r = 200
        d = D(geo='s',nz=300,nx=round(np.pi*200*2),method='solve4',sparse=True,r=r,lx=np.pi*r,xhot=(r+150)*0.5,mbase='F', thot=1e8)
        d.plot()

        # two poles
        r = 200
        d = D(geo='s',nz=300,nx=round(np.pi*200),method='solve4',sparse=True,r=r,lx=0.5*np.pi*r,xhot=(r+150)*0.5,mbase='F', thot=1e8)
        d.plot(mirror=4)

        # equator
        r = 200
        d = D(geo='e',nz=300,nx=round(np.pi*200),method='solve4',sparse=True,r=r,lx=0.5*np.pi*r,xhot=(r+150)*0.5,mbase='F', thot=1e8)
        d.plot(mirror=4)

        # sphere
        r = 1
        d = D(
            geo='s',
            nz=300,
            nx=round(np.pi*200*2),
            method='solve4',
            sparse=True,
            r=r,
            lx=r*np.pi,
            xhot=(r + 150)*0.1,
            thot=1e8,
            mbase='F',
            fbase=0,
            )
        d.plot()

        # bend
        r = 1000
        rmin = 200
        res = 0.5*10
        h = 150
        w = np.pi*rmin
        xhot = 20
        d = D(
            geo='s',
            nz=round(h/res),
            nx=round(w/res),
            method='solve4',
            r=r,
            lx=w,
            lz=h,
            xhot=xhot*(r + h)/r,
            thot=1e8,
            mbase='F',
            )
        d.plot(vmax=1e8,vmin=3e6)

        # T evolution movie
        e = D(nx=80, nz=30, xhot=0, thot=1e8)
        e.xhot=20
        e.evo(
            movie='T1e8x20.webm',
            tstop=86400,
            pkwargs=dict(vmin=4e6,vmax=1e8),
            fkwargs=dict(size=(800,320)),
            store=True,
            )

        # burn movie
        e = D(geo='c', nx=80, nz=30, xhot=0, thot=2e9, method='solve4', net='fancy')
        e.xhot=100
        e.evo(dTr=1,dt0=1e-8,tstop=86400)
        e.store2movie(
            #'T2e9x100cX.mp4',
            'y.mp4',
            mkwargs=dict(delay=1/60),
            fkwargs=dict(size=(800,400)),
            pkwargs=dict(
                q=('c12', 'mg24'),
                pad = (0.15, 0.2),
                vmin=(0,0),vmax=(1,1)),
            sel=slice(None, None, 1))
        e.store2movie(
            'T2e9x100ceT.mp4',
            mkwargs=dict(delay=1/60),
            fkwargs=dict(size=(800,400)),
            pkwargs=dict(
                vmin=(7e6,1e10),
                pad = (0.15, 0.2),
                vmax=(2e9,5e17)),
            sel=slice(None, None, 1))


def bend_movie(filename='bend.webm', fps=60, res=5, steps=600, mp=False, **kwargs):
    rmax = 1e6
    rmin = 200
    rr = rmax/(np.arange(steps)/(steps-1)*(rmax/rmin - 1) + 1)
    h = 150
    w = np.pi * rmin

    fkwargs = dict(
        xhot = 20,
        thot = 1e8,
        mbase = 'F',
        )
    fkwargs.update(kwargs)


    def generator(r, fig):
        d = Diffuse(
            geo = 's',
            nz = round(h / res),
            nx = round(w * (h + r) / (r * res)),
            method = 'solve4',
            r = r,
            lx = w,
            lz = h,
            **fkwargs,
            )
        d.plot(
            vmax = 1e8,
            vmin = 3e6,
            xlim = [-w * 1.1, w * 1.1],
            ylim = [-(2 * rmin + h) * 1.1, h * 1.1],
            fig = fig,
            pos =   [0.1, 0.3, 0.875, 0.65],
            cbpos = [0.1, 0.1, 0.875, 0.1 ],
            )

    make_movie(
        filename,
        generator = generator,
        canvas = MPLCanvas,
        data = 'r',
        values = rr,
        fps = fps,
        mp = mp,
        )

def hole_movie(filename='hole.webm', geo='s', fps=60, res=5, steps=600, xhotf=0.1, mp=False, **kwargs):
    rmax = 1e6
    rmin = .1
    rr = np.exp(np.arange(steps) / (steps-1) * (np.log(rmin) - np.log(rmax))) * rmax
    h = 150

    fkwargs = dict(
        thot = 1e8,
        mbase = 'F',
        geo = geo,
        )
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
            nz = round(h / res),
            nx = round(w * 200 / (r * res)),
            method = 'solve4',
            r = r,
            lx = w,
            lz = h,
            xhot = x * xhotf,
            **fkwargs,
            )
        d.plot(
            vmax = 1e8,
            vmin = 3e6,
            xlim = 1.05 * x * np.array([-1, 1]) ,
            ylim = 1.05 * x * np.array([-1, 1]) - r,
            fig = fig,
            pos =   [0.1, 0.2  , 0.875, 0.75],
            cbpos = [0.1, 0.025, 0.875, 0.1],
            **pkwargs,
            )

    make_movie(
        filename,
        generator = generator,
        canvas = MPLCanvas,
        ckwargs = dict(size=(600,720)),
        data = 'r',
        values = rr,
        fps = fps,
        mp = mp,
        )
