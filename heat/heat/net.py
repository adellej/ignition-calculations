import numpy as np
import numpy.linalg

from isotope import ion as I

from numeric import *

class Net(object):
    q = 0

    ions = list()

    A = list()

    def massfrac(self, ion):
        for i,x in enumerate(self.ions):
            if x == ion:
                return self.ppn[..., i] * self.A[i]

    def molfrac(self, ion):
        for i,x in enumerate(self.ions):
            if x == ion:
                return self.ppn[..., i]

    def __init__(self, ppn, shape, stationary = False):
        self.shape = shape

        if shape is None:
            self.ppn = ppn
        else:
            ppn = np.atleast_1d(ppn)
            self.ppn = np.tile(ppn, shape + (1,))

    def revert_update(self):
        if hasattr(self, '_ppn'):
            self.ppn = self._ppn.copy()

    def checkpoint(self):
        self._ppn = self.ppn.copy()

    def rate(self, tem, den):
        raise NotImplementedError()

    def eps(self, tem, den):
        return self.q * self.rate(tem, den)

    def eps_T(self, tem, den, dtr = 1e-7):
        eps = self.eps(tem, den)
        dtem = dtr * tem
        epp = self.eps(tem + dtem, den)
        epm = self.eps(tem - dtem, den)
        depsdT = (epp - epm) / (2 * dtem)

    def epsdt(self, tem, den, dt, update = False):
        """
        default: ignore time evolution
        """
        return self.eps(tem, den)

    def epsdt4(self, tem, den, dt, update = False):
        """
        default: ignore time evolution
        """
        return self.eps(qqrt(tem), den)

    def epsdt_T(self, tem, den, dt, dtr = 1e-7):
        dtem = dtr * tem
        epp = self.epsdt(tem + dtem, den, dt)
        epm = self.epsdt(tem - dtem, den, dt)
        eps = self.epsdt(tem, den, dt)
        deps = (epp - epm) / (2 * dtem)
        return eps, deps

    def eps4_T4(self, tem4, den):
        tem = qqrt(tem4)
        eps, deps = self.eps(tem, den)
        deps = 0.25 * deps / cube(tem)
        return eps, deps

    def epsdt4_T4(self, tem4, den, dt):
        tem = qqrt(tem4)
        eps, deps = self.epsdt_T(tem, den, dt)
        deps = 0.25 * deps / cube(tem)
        return eps, deps


class NetNone(Net):
    """
    no net drop-in
    """
    def rate(self, tem, den):
        return np.zeros_like(tem)

class Net3aFFNs(Net):
    """
    'static' network only depending on helium abundace

    Rate from KWW12 (FFN)

    factor 4.64155e-6 = 5.09e11 * 4**3 / q

    q is energy per mol

    """

    q = 7.274 * 1.602176634e-06 * 6.02214076e+23

    ions = I(['he4', 'c12'])

    A = np.array([i.A for i in ions])

    @staticmethod
    def lam(tem):
        """
        from KWW12 (FFN)
        """
        t8i = 1e8 / tem
        # neglect screening (f3a)
        f = 4.64155e-6 * cube(t8i) * np.exp(-44.027 * t8i)
        return f

    def rate(self, tem, den):
        return self.lam(tem) * den**2 * cube(self.ppn[..., 0])

class Net3aAngulo99s(Net3aFFNs):
    """
    'static' network only depending on helium abundace

    rate from Angulo99

    factor 5.7194e-17 = 6.272 * 4**3 / q
    """
    def lam(tem):
        t9 = 1e-9 * tem
        t9m1 = 1 / t9
        t92 = t9**2
        t913 = cbrt(t9)
        t9m13 = 1 / t913
        t923 = t913**2
        t9m23 = 1 / t923
        t9m32 = np.sqrt(cube(t9m1))
        exp = np.exp
        r = (5.7194e-17 *
             (1 + 0.0158 * t9**(-0.65)) *
             (2.43e9 * t9m23 * exp (-13.490 * t9m13 - t92 / 0.025) * (1 + 74.5 * t9)
              + 6.09e5 * t9m32 * exp(-1.054 * t9m1)) *
             (2.76e7 * t9m23 * exp(-23.570 * t9m13 - t92 * 6.25) * (1 + 5.47 * t9 + 326 * t92)
              + 130.7 * t9m32 * exp(-3.338 * t9m1)
              + 2.51e4 * t9m32 * exp(-20.307 * t9m1))
            )
        return r

class Net3aFFNr(Net3aFFNs):
    def xlamr(self, tem, den):
        """
        from KWW12 (FFN)
        """
        f = self.lam(tem)
        r = f * 1.2e-6 * cube(tem) * np.exp(-8.424e10 / tem)
        return f * den**2, r

    def rate(self, tem, den):
        f, r = self.xlamr(tem, den)
        return f * cube(self.ppn[..., 0]) - r * self.ppn[..., 1]

class Net3aFFNt(Net3aFFNr):

    ymin = 1e-2

    def epsdt(self, tem, den, dt, update = False):
        if dt is None:
            return self.eps(tem, den)
        f, r = self.xlamr(tem, den)
        a2 = self.ppn[..., 0]**2
        a3 = a2 * self.ppn[..., 0]
        c = self.ppn[..., 1]
        dti = 1 / dt
        b = np.moveaxis(np.array(
            (
                - 3 * f * a3 + 3 * r * c,
                      f * a3     - r * c,
            )), 0, -1)

        A = np.moveaxis(np.array(
            (
                (dti + 9 * f * a2,     - 3 * r),
                (    - 3 * f * a2, dti +     r),
            )), (0,1), (-2,-1))

        dy = np.linalg.solve(A, b)
        eps = self.q * dy[..., 1] * dti
        if update:
            self._ppn = self.ppn.copy()
            self.ppn = self.ppn + dy
            self.dt = np.min((self.ppn + self.ymin) / (np.abs(dy) + 1e-99))

        return eps


class NetVec(Net):

    q = np.atleast_1d(0)

    def eps(self, tem, den):
        return np.sum(self.q * self.rate(tem, den), axis=-1)


class Net3aC12(NetVec):

    q = np.array([ 7.275, 13.933]) * 1.602176634e-06 * 6.02214076e+23

    me = np.array([0, - q[0] / 12, - q[0] / 12 + q[1] / 24])

    ions = I(['he4', 'c12', 'mg24'])

    A = np.array([i.A for i in ions])

    ymin = 1e-2

    @staticmethod
    def r_3a(t9, rho = 1, sc = 1):
        """Compute lambda for 3a rate and its reverse"""
        # q = 7.275 MeV
        t9m1 = 1 / t9
        t9log = np.log(t9)
        t913 = np.exp(t9log * (1 / 3))
        t9m13 = 1 / t913
        t953 = t9 * t913**2
        t93 = cube(t9)
        rho2 = rho**2
        ra3 = (
            + np.exp(
                -9.710520e-01
                -3.706000e+01 * t9m13
                +2.934930e+01 * t913
                -1.155070e+02 * t9
                -1.000000e+01 * t953
                -1.333330e+00 * t9log)
            + np.exp(
                -2.435050e+01
                -4.126560e+00 * t9m1
                -1.349000e+01 * t9m13
                +2.142590e+01 * t913
                -1.347690e+00 * t9
                +8.798160e-02 * t953
                -1.316530e+01 * t9log)
            + np.exp(
                -1.178840e+01
                -1.024460e+00 * t9m1
                -2.357000e+01 * t9m13
                +2.048860e+01 * t913
                -1.298820e+01 * t9
                -2.000000e+01 * t953
                -2.166670e+00 * t9log))
        f = ra3 * rho2 * (sc / 6)
        rev = 2.00e20 * t93 * np.exp(-84.424 * t9m1)
        r = rev * ra3
        return f, r

    @staticmethod
    def r_2c(t9, rho = 1, sc = 1):
        """compute lambda for C12+C12 reaction and its inverse"""
        # q = 13.933 MeV
        t9a = t9 / (1 + 0.0396 * t9)
        t9am13 = 1 / cbrt(t9a)
        t9a56 = t9a * t9am13
        t932 = t9 * np.sqrt(t9)
        t9m32 = 1 / t932
        t9m1 = 1 / t9
        t93 = t932**2
        r24 = (4.27e+26 * t9a56 * t9m32 *
               np.exp(-84.165 * t9am13 - 0.00212 * t93))
        f = (0.5 * sc) * rho * r24
        rev = 7.2517952463e10 * t932 * np.exp(-161.6858 * t9m1)
        r = rev * r24
        return f, r

    def rate(self, tem, den):
        t9 = tem * 1e-9
        fa, ra = self.r_3a(t9, den)
        fc, rc = self.r_2c(t9, den)
        return  np.moveaxis(np.array((
            fa * cube(self.ppn[..., 0]) - ra * self.ppn[..., 1],
            fc * self.ppn[..., 1]**2 - rc * self.ppn[..., 2],
            )), 0, -1)

    def epsdt(self, tem, den, dt, update = False):
        if dt is None:
            return self.eps(tem, den)
        dti = 1 / dt
        dy = self.netsolve(tem, den, dti)
        eps = - np.sum(self.me * dy, axis=-1) * dti
        if update:
            self.update(dy=dy)
        return eps

    def netsolve(self, tem, den, dti):
        t9 = tem * 1e-9
        fa, ra = self.r_3a(t9, den)
        fc, rc = self.r_2c(t9, den)
        a2 = self.ppn[..., 0]**2
        a3 = a2 * self.ppn[..., 0]
        c = self.ppn[..., 1]
        c2 = c**2
        m = self.ppn[..., 2]
        z = np.zeros_like(tem)
        b = np.moveaxis(np.array(
            (
                - 3 * fa * a3 + 3 * ra * c,
                      fa * a3 -     ra * c - 2 * fc * c2 + 2 * rc * m,
                                                 fc * c2 -     rc * m,
            )), 0, -1)

        A = np.moveaxis(np.array(
            (
                (dti + 9 * fa * a2,     - 3 * ra             ,            z),
                (    - 3 * fa * a2, dti +     ra + 4 * fc * c,     - 2 * rc),
                (                z,              - 2 * fc * c, dti +     rc),
            )), (0,1), (-2,-1))

        dy = np.linalg.solve(A, b)
        return dy

    def update(self, den=None, tem=None, dt=None, dy=None):
         self._ppn = self.ppn.copy()
         if dy is None:
             dti = 1 / dt
             dy = netsolve(self, tem, den, dti)
         self.ppn = self.ppn + dy
         self.dt = np.min((self.ppn + self.ymin) / (np.abs(dy) + 1e-99))
