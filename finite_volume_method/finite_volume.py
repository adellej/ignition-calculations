''' This code solves a 2D heat equation using the finite volume method for a hotspot on the surface of an accreting neutron star.

To run, in python type:

from finite_volume import Diffuse

D = Diffuse(
    Nx=round(2*3.1415*200),
    Nz=300,
    Lx=3.1415*100,
    Lz=200,
    Tcrust=1e8,
    Thotspot=1e8,
    Ttop=1e6,
    Fbase=1.0e19,
    hsize=100,
    tol=1e-6,
    maxIter=500,
    op=1.0,
    run_name='test',
    rhoflag='fit',
    coord="cylindrical"
)
To run:
D.run()
'''

import numpy as np
from numpy import sqrt
from scipy import integrate
from scipy import interpolate
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import scipy.linalg
import color
import cmasher as cmr
from kepdump import load as D


class Diffuse:
    def __init__(
        self,
        Nx=80,
        Nz=80,
        Lx=600,
        Lz=600,
        Tcrust=1e7,
        Thotspot=5e7,
        Ttop=1e6,
        bb_flag='bb',
        Fbase=1.0e21,
        hsize=20,
        tol=1e-6,
        op=1.0,
        maxIter=300,
        run_name="ign",
        rhoflag="fit",
        coord="cylindrical",
        kappa_path='/Users/adelle/Documents/ignition_calculations/cylindrical_coords/finite_volume_method/kappa.npz',
        dump_path='/Users/adelle/Documents/ignition_calculations/cylindrical_coords/Alex/heat/dumps/heat2#240',
        substrate = True,
    ):
        import numpy as np

        # define size of mesh:
        self.Nx = Nx
        #self.Nz = Nz
        self.Lx = Lx  # cm
        # self.Lz = Lz  # cm
        # define scaling between physical units and grid size for x and z:

        # temperatures in K:
        self.Thotspot = Thotspot

        # define sself.Ize of hotspot:
        self.hsize = hsize

        # numerical parameters:
        self.tol = tol
        self.op = op
        self.maxIter = maxIter

        self.substrate = substrate

        # option for density, either fit ('fit'), completely degenerate ('deg') or constant ('const'):
        if dump_path is not None:
            x0 = 0
            z0 = 0
            dump = D(dump_path)
            j0 = np.maximum(dump.jshell0, dump.jburnmin)
            if substrate:
                jsub = j0
                j0 = 1
            else:
                jsub = 0
            j1 = dump.jm
            sel = slice(j0, j1 + 1)
            #sel = slice(40, j1 + 1)
            dz1 = dump.dr[sel]
            nz = len(dz1)
            self.Nz = nz
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

            # if j0 > 0:
            #     tbase = sqrt(dump.tn[j0] * dump.tn[j0 - 1])
            # else:
            #     tbase = dump.tn[j0]
            # if j0 == 1:
            #     fbase = dump.xlum0
            # else:
            #     fbase = dump.xln[j0 - 1]
            # fbase /= 4 * np.pi * dump.rn[j0 - 1] ** 2
            # ttop = dump.teff
            self.rhoflag = 'dump'
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
            lz = zf1[-1] - zf1[0]
            self.Lz = lz

            self.dump = dump
            self.jsub = jsub
            self.tc1 = dump.tn[sel]
            self.dump = dump
            self.Fbase = fbase
            self.Tcrust = tbase
            self.Ttop = ttop
            # self.Ttop = self.tc1[-1]
            # self.z = zc1
            self.z = zf1
            self.dz = []
            # for i in range(len(zc1)-1):
            #     self.dz.append(zc1[i+1] - zc1[i])
            # self.dz.append(self.dz[-1])
            for i in range(len(zf1) - 1):
                self.dz.append(zf1[i + 1] - zf1[i])
            self.dz.append(self.dz[-1])
            #
            # rhoc = np.tile(rhoc1, (Nx, Nz, 1))
            # # rhoci = np.tile(rhoci1, (Nx , Nz , 1))
            #
            # rhoz = np.tile(rhof1, (Nx, Nz, 1))
            # # rhozi = np.tile(rhofi1, (Nx , Nz , 1))
            #
            # rhox = np.tile(rhoc1, (Nx, Nz, 1))
            # rhoxi = np.tile(rhoci1, (Nx, Nz , 1))

            self.rhoc = rhoc1
            self.rhox = rhof1
            self.rhoz = rhof1

            print(self.rhoc[0], self.rhoz[0])
            print(f'Height is: {self.Lz}')

            plt.plot(self.z, self.rhoz)
            plt.yscale('log')
            plt.savefig('density_init.pdf')

        else:
            self.rhoflag = rhoflag
            self.Fbase = Fbase
            self.Tcrust = Tcrust
            self.Ttop = Ttop
            compz1 = np.full(nz, 0)
            self.Nz = Nz
            self.Lz = Lz  # cm



            # x0 = 0
            # z0 = 0
            # self.rhoflag = 'dump'
            # dump = D(dump_path)
            # j0 = np.maximum(dump.jshell0, dump.jburnmin)
            # j1 = dump.jm
            # sel = slice(j0, j1 + 1)
            # dz1 = dump.dr[sel]
            # nz = len(dz1)
            # zres = 'dump'
            # self.Nz = nz
            # zf1 = np.ndarray(self.Nz + 1)
            # zf1[0] = z0
            # zf1[1:] = z0 + np.cumsum(dz1)
            # zc1 = 0.5 * (zf1[1:] + zf1[:-1])
            # self.r = dump.rn[j0 - 1]
            # tbase = 0.5 * (dump.tn[j0] + dump.tn[j0 - 1])
            # if j0 == 1:
            #     fbase = dump.xlum0
            # else:
            #     fbase = dump.xln[j0 - 1]
            # fbase /= 4 * np.pi * dump.rn[j0 - 1] ** 2
            # ttop = dump.teff
            # rhoflag = 'dump'
            # rhoc1 = dump.dn[sel]
            # rhof1 = np.ndarray((nz + 1,))
            # rhof1[1:-1] = 0.5 * (rhoc1[1:] + rhoc1[:-1])
            # if j0 > 0:
            #     rhof1[0] = 0.5 * (rhoc1[j0] + rhoc1[j0 - 1])
            # else:
            #     rhof1[0] = rhoc[0]
            # if j1 >= dump.jm:
            #     rhof1[-1] = rhoc1[-1]
            # else:
            #     rhof1[-1] = 0.5 * (rhoc1[j1] + rhoc1[j1 + 1])
            # self.tc1 = dump.tn[sel]
            # self.dump = dump
            # self.Fbase = fbase
            # self.Tcrust = tbase
            # self.Ttop = ttop
            # # self.Ttop = self.tc1[-1]
            # # self.z = zc1
            # self.z = zf1
            # self.dz = []
            # # for i in range(len(zc1)-1):
            # #     self.dz.append(zc1[i+1] - zc1[i])
            # # self.dz.append(self.dz[-1])
            # for i in range(len(zf1) - 1):
            #     self.dz.append(zf1[i + 1] - zf1[i])
            # self.dz.append(self.dz[-1])

            # f = plt.figure(figsize=(10,8))
            # plt.plot(self.z, rhoc1)
            # plt.plot(self.z, rhof1[:-1])
            # plt.show()
            # plt.close()

            # make density grid
            # rhofi1 = 1 / rhof1
            # rhoci1 = 1 / rhoc1

        #     rhoc = np.tile(rhoc1, (Nx, Nz, 1))
        #     # rhoci = np.tile(rhoci1, (Nx , Nz , 1))
        #
        #     rhoz = np.tile(rhof1, (Nx, Nz, 1))
        #     # rhozi = np.tile(rhofi1, (Nx , Nz , 1))
        #
        #     rhox = np.tile(rhoc1, (Nx, Nz, 1))
        #     # rhoxi = np.tile(rhoci1, (Nx, Nz , 1))
        #
        #     self.rhoc = rhoc1
        #     self.rhox = rhof1
        #     self.rhoz = rhof1
        #     print(self.rhoc[0], self.rhoz[0])
        #
        # else:
        #     self.rhoflag = rhoflag
        #     self.Fbase = Fbase
        #     self.Tcrust = Tcrust
        #     self.Ttop = Ttop

        # choose name to save output files under
        self.run_name = run_name

        # constants:
        self.const_kappa = 0.136  # cm^2/g, assume constant opacity
        self.a = 7.566e-15  # radiation constant
        self.c = 2.998e10  # speed of light
        self.kb = 1.231e15 * (2 ** (4 / 3))  # ideal gas constant for pure helium
        self.g = 2.426e14  # surface gravity, cgs
        self.Y = 0.99  # helium mass fraction of material
        self.mu = 4 / 3  # mu
        self.sigma = 5.6704e-5  # Steffan Boltzmann constant

        # get kappa interpolation from table:
        if substrate:
            #load iron comp kappa table
            self.kappa_path_Fe = kappa_path[1]
            data = np.load(self.kappa_path_Fe, allow_pickle=True)
            tem = np.power(data['tem'], 4)
            den = data['den']
            kappa = data['kappa']
            self.kappa_t4_Fe = interpolate.RectBivariateSpline(
                tem, den, kappa, kx=3, ky=3, s=1
            )
            self.kappa_limits_Fe = np.array([[min(tem), max(tem)], [min(den), max(den)]])

            # load helium comp kappa table
            self.kappa_path = kappa_path[0]
            data = np.load(self.kappa_path, allow_pickle=True)
            tem = np.power(data['tem'], 4)
            den = data['den']
            kappa = data['kappa']
            self.kappa_t4 = interpolate.RectBivariateSpline(
                tem, den, kappa, kx=3, ky=3, s=1
            )
            self.kappa_limits = np.array([[min(tem), max(tem)], [min(den), max(den)]])

        else:
            self.kappa_path = kappa_path[0]
            data = np.load(self.kappa_path, allow_pickle=True)
            tem = np.power(data['tem'], 4)
            den = data['den']
            kappa = data['kappa']
            self.kappa_t4 = interpolate.RectBivariateSpline(
                tem, den, kappa, kx=2, ky=2, s=1
            )
            self.kappa_limits = np.array([[min(tem), max(tem)], [min(den), max(den)]])

        # T = np.maximum(self.kappa_limits[0,0], np.minimum(self.kappa_limits[0,1], T4))
        # rho = np.maximum(self.kappa_limits[1,0], np.minimum(self.kappa_limits[1,1], rhoin))

        # initalise mesh:
        # make mesh:

        self.x = np.linspace(0, self.Lx, self.Nx + 1)
        if dump_path is None or '':
            self.z = np.linspace(0, self.Lz, self.Nz + 1)
            self.dz = self.z[1] - self.z[0]
        self.dx = self.x[1] - self.x[0]

        # introduce index sets for all points:
        self.Ix = range(0, self.Nx + 1)
        self.Iz = range(0, self.Nz + 1)

        # define mapping to fill matrices later:
        # define mapping from 2D to 1D matrix:
        # self.m = lambda i, j: j * (self.Nx + 1) + i
        self.m = lambda i, j: j * self.Nx + i

        self.coord = coord
        self.bb_flag = bb_flag

        # -------------------------------------------------------------------------------------------------- #

        # define function for areas for different coordinate systems:

    def get_a_e(self, x_i, j):
        if self.coord == "cylindrical":
            if self.rhoflag == 'dump':
                if j == len(self.dz):
                    return self.dz[j - 1] * 2 * np.pi * x_i
                else:
                    return self.dz[j] * 2 * np.pi * x_i
            else:
                return self.dz * 2 * np.pi * x_i
        if self.coord == "cartesian":
            return self.dz

    def get_a_s(self, x_f, x_i):
        if self.coord == "cylindrical":
            return np.pi * (x_f ** 2 - x_i ** 2)
        if self.coord == "cartesian":
            return self.dx

    def get_h(self, j):
        if self.coord == "cylindrical":
            if self.rhoflag == 'dump':
                if j == len(self.dz):
                    return self.dz[j - 1]
                else:
                    return self.dz[j]
            else:
                return self.dz
        if self.coord == "cartesian":
            return self.dx

        # -------------------------------------------------------------------------------------------------- #

    # define density and heating functions:

    def rho(self, zcm):
        """this function returns the density. If self.rhoflag = deg, then it returns the density for a completely degenerate equation of state. If self.rhoflag = 'fit' it returns a density fit taken from fitting the rho(self.z) profile for a 1D model from Bildsten 1998 written by Frank. If self.rhoflag = 'const', it returns constant density of rho=1e5.
        # (profile also agrees with Andrew Cumming's settle code) """

        rho0 = np.exp(1.253e1)
        beta = -4.85e-10
        alpha = 4.435
        H = self.z[-1]
        rho_t = 10  # g/cm^3

        if self.rhoflag == "deg":
            return (
                rho_t * (1.0 + self.g / (4 * self.kb * rho_t ** (1 / 3)) * (H - zcm))
            ) ** 3
        elif self.rhoflag == "fit":
            return rho0 * np.exp(beta * np.power(zcm, alpha))
        elif self.rhoflag == "const":
            return 1e5  # option for constant density
        elif self.rhoflag == "settle":
            a = 2.2007645e07
            b = 9.1394875e-01
            # density function returns 0 at z=0 so return 1 instead here
            # if zcm == 0:
            #     return 1.0
            # else:
            return a * np.power(b, zcm)
        elif self.rhoflag == "kepler":
            a = -7.18978907e04
            b = 4.98392634e-01
            c = 1.69570981e06
            return a * zcm ** b + c

    def logrhoprime(self, zcm):
        """function to calculate log derivative of rho(self.z) """

        beta = -4.85e-10
        alpha = 4.435
        if self.rhoflag == "deg":
            return -3 * self.g / (4 * self.kb) * np.array(self.rho(zcm)) ** (-1 / 3)
        elif self.rhoflag == "fit":
            return beta * alpha * np.power(zcm, alpha - 1)
        elif self.rhoflag == "const":
            return 0.0  # option for constant density

    def get_kappa(self, T4, rhoin, j):
        """function to read opacity table and extract kappa for given T**4 and rho"""

        if self.substrate:
            if j < self.jsub:
                f = self.kappa_t4_Fe
                T = np.maximum(self.kappa_limits_Fe[0, 0], np.minimum(self.kappa_limits_Fe[0, 1], T4))
                rho = np.maximum(
                    self.kappa_limits_Fe[1, 0], np.minimum(self.kappa_limits_Fe[1, 1], rhoin)
                )
                kappa = f(T, rho)
                dkappa = f(T, rho, dx=1, dy=0, grid=False)
                #print(f'j = {j}, calling Fe kap table')
            else:
                f = self.kappa_t4
                T = np.maximum(self.kappa_limits[0, 0], np.minimum(self.kappa_limits[0, 1], T4))
                rho = np.maximum(
                    self.kappa_limits[1, 0], np.minimum(self.kappa_limits[1, 1], rhoin)
                )
                kappa = f(T, rho)
                dkappa = f(T, rho, dx=1, dy=0, grid=False)
        else:
            f = self.kappa_t4
            T = np.maximum(self.kappa_limits[0, 0], np.minimum(self.kappa_limits[0, 1], T4))
            rho = np.maximum(
                self.kappa_limits[1, 0], np.minimum(self.kappa_limits[1, 1], rhoin)
            )
            kappa = f(T, rho)
            dkappa = f(T, rho, dx=1, dy=0, grid=False)

        #print(f'kappa = {kappa[0, 0]}')
        return kappa[0, 0]

    # define eps heating due to nuclear burning function:
    def eps(self, T, density , j):
        """ heating due to nuclear nurning, assuming triple alpha reactions only, from Bildsten 1998 eqn 16 """

        T8 = np.power(T,0.25) / 1e8
        rho5 = density / 1e5

        if self.substrate:
            if j < self.jsub :
                res = 0.0
            else:
                Y = 0.99
                res = (5.3e21 * rho5 ** 2 * Y ** 3 / T8 ** 3) * np.exp(-44 / T8)
        else:
            Y = 0.99  # helium mass fraction
            res = (5.3e21 * rho5 ** 2 * Y ** 3 / T8 ** 3) * np.exp(-44 / T8)

        return res  # in erg/g/s
        #return 0.0  #option to switch off nuclear burning

    # Now fill T initial

    def get_Tsides(self, Tbase4, Tsurface4, zi):
        """ function to get analytic solution for temperature as a function of depth"""

        zbase = 0
        zsurf = zi[-1]

        rhoint = integrate.quad(self.rho, a=zbase, b=zsurf)
        c = (Tbase4 - Tsurface4) / rhoint[0]

        rhointz = []
        for i in range(len(zi)):
            integ = integrate.quad(self.rho, a=zbase, b=zi[i])
            rhointz.append(integ[0])

        T = Tsurface4 + c * np.array(rhointz)

        return 0.5 * (T[1:] + T[:-1])

    # load initial conditions into T_n:
    def fill_Tinitial(self):
        """ function to fill the initial conditions of temperature """

        # get self.x indices of hotspot location (middle)
        # hotspoty = np.arange(int((self.Nx+1)/2 - self.hsize/2),int((self.Nx+1)/2 + self.hsize/2))
        hotspoty = np.arange(0, int(self.hsize / 2))

        # initialise T array:
        Tint = np.zeros((self.Nx, self.Nz))
        if self.rhoflag == 'dump':
            for i in range(self.Nx):
                Tint[i, :] = self.tc1 ** 4
            j = self.Nz - 1
            if self.hsize != 0:
                for i in hotspoty:
                    Tint[i, j] = self.Thotspot ** 4

        else:
            # we solve for T**4 so need to convert our inputs in (K)
            Ttop4 = self.Ttop ** 4
            Tcrust4 = self.Tcrust ** 4
            Thotspot4 = self.Thotspot ** 4

            # do crust (bottom):
            j = 0
            for i in range(self.Nx):
                Tint[i, j] = Tcrust4

            # fill with analytic solution, don't need to worry about setting sides differently as we have periodic boundary conditions:
            for i in range(self.Nx):
                Tint[i, :] = self.get_Tsides(Tcrust4, Ttop4, self.z)[::-1]

            # now do surface:
            j = self.Nz - 1
            for i in range(self.Nx):
                Tint[i, j] = Ttop4
            for i in hotspoty:
                Tint[i, j] = Thotspot4

        return Tint

    # -------------------------------------------------------------------------------------------------- #

    # we set up to solve as a matrix system, with A*c = b where c is delta T**4, some small increase in T until we get a solution.
    def fill_A(self, T, A):
        """ fill A coefficient matrix for solving system A*c = b """

        # what should Rin and Rout be for weighting for north and south points?? Total R of cylinder, or R of zone?
        for j in range(self.Nz):
            for i in range(self.Nx):  # i.e. i = 1:79
                p = self.m(i, j)

                if i == 0:
                    a_w = 0.0
                else:
                    a_w = self.get_a_e(self.x[i], j) / (
                        self.rho(self.z[j])
                        * self.get_kappa(T[i, j], self.rho(self.z[j]))
                        * self.dx
                    )
                    A[p, self.m(i - 1, j)] = a_w

                if i == self.Nx - 1:
                    ip1 = self.Nx - 1
                    a_e = 0.0
                else:
                    ip1 = i + 1
                    a_e = self.get_a_e(self.x[i + 1], j) / (
                        self.rho(self.z[j])
                        * self.get_kappa(T[i, j], self.rho(self.z[j]))
                        * self.dx
                    )
                    A[p, self.m(i + 1, j)] = a_e
                if j == 0:
                    # a_s = 0.0
                    a_s = self.get_a_s(self.x[i + 1], self.x[i]) / (
                        self.rho(self.z[j])
                        * self.get_kappa(T[i, j], self.rho(self.z[j]))
                        * self.dz
                    )
                else:
                    a_s = self.get_a_s(self.x[i + 1], self.x[i]) / (
                        self.rho(self.z[j])
                        * self.get_kappa(T[i, j], self.rho(self.z[j]))
                        * self.dz
                    )
                    A[p, self.m(i, j - 1)] = a_s

                if j == self.Nz - 1:
                    hotspoty = np.arange(0, int(self.hsize / 2))
                    if self.bb_flag == 'bb' and i not in hotspoty:
                        # a_n = 3 * self.sigma * self.Ttop**3 * self.get_a_s(self.x[i+1], self.x[i])
                        a_n = 0.0
                    # a_n = (
                    # self.sigma
                    # * (T[i, j] + self.Ttop ** 4)
                    # * 0.5
                    # * self.get_a_s(self.x[i + 1], self.x[i])
                    # )
                    else:
                        a_n = self.get_a_s(self.x[i + 1], self.x[i]) / (
                            self.rho(self.z[j])
                            * self.get_kappa(T[i, j], self.rho(self.z[j]))
                            * self.dz
                        )

                else:
                    a_n = self.get_a_s(self.x[i + 1], self.x[i]) / (
                        (self.rho(self.z[j + 1]))
                        * self.get_kappa(T[i, j], self.rho(self.z[j]))
                        * self.dz
                    )
                    A[p, self.m(i, j + 1)] = a_n

                a_p = -a_e - a_w - a_s - a_n

                A[p, p] = a_p

        return A

    # define function to fill b, the source term matrix:
    def fill_b(self, T, b):
        """ fill b matrix for solving A*c = b """

        # Compute b
        for j in range(self.Nz):
            for i in range(self.Nx):  # i.e. i = 1:79

                if i == 0:
                    im1 = 0
                    a_w = 0
                else:
                    im1 = i - 1
                    a_w = self.get_a_e(self.x[i], j) / (
                        self.rho(self.z[j])
                        * self.get_kappa(T[i, j], self.rho(self.z[j]))
                        * self.dx
                    )

                if i == self.Nx - 1:
                    ip1 = self.Nx - 1
                else:
                    ip1 = i + 1
                if j == 0:
                    a_s = 0.0
                    # a_s_2 = self.Fbase * self.get_a_s(self.x[i + 1], self.x[i])
                    # a_s = self.get_a_s(self.x[i + 1], self.x[i]) / (
                    #     self.rho(self.z[j])
                    #     * self.get_kappa(T[i, j], self.rho(self.z[j]))
                    #     * self.dz
                    # )
                    a_s_2 = a_s * self.Tcrust ** 4

                else:
                    a_s = self.get_a_s(self.x[i + 1], self.x[i]) / (
                        self.rho(self.z[j])
                        * self.get_kappa(T[i, j], self.rho(self.z[j]))
                        * self.dz
                    )
                    a_s_2 = a_s * T[i, j - 1]

                if j == self.Nz - 1:
                    hotspoty = np.arange(0, int(self.hsize / 2))
                    if i in hotspoty:
                        a_n = self.get_a_s(self.x[i + 1], self.x[i]) / (
                            self.rho(self.z[j])
                            * self.get_kappa(T[i, j], self.rho(self.z[j]))
                            * self.dz
                        )
                        a_n_2 = a_n * self.Thotspot ** 4

                    else:
                        if self.bb_flag == 'bb':
                            # a_n = self.sigma * self.get_a_s(self.x[i+1], self.x[i])
                            # a_n = 0.0
                            # a_n_2 = (
                            #     self.sigma
                            #     * self.Ttop ** 4
                            #     * self.get_a_s(self.x[i + 1], self.x[i])
                            # )

                            # a_n = self.sigma * self.get_a_s(self.x[i + 1], self.x[i])
                            a_n = 0.0
                            a_n_2 = (
                                self.sigma
                                * (self.Ttop ** 4)
                                * self.get_a_s(self.x[i + 1], self.x[i])
                            )

                        else:
                            a_n = self.get_a_s(self.x[i + 1], self.x[i]) / (
                                self.rho(self.z[j])
                                * self.get_kappa(T[i, j], self.rho(self.z[j]))
                                * self.dz
                            )
                            a_n_2 = a_n * self.Ttop ** 4

                else:
                    a_n = self.get_a_s(self.x[i + 1], self.x[i]) / (
                        (self.rho(self.z[j + 1]))
                        * self.get_kappa(T[i, j], self.rho(self.z[j]))
                        * self.dz
                    )
                    a_n_2 = a_n * T[i, j + 1]

                a_e = self.get_a_e(self.x[i + 1], j) / (
                    self.rho(self.z[j])
                    * self.get_kappa(T[i, j], self.rho(self.z[j]))
                    * self.dx
                )

                S = (
                    self.rho(self.z[j])
                    * (self.get_h(j))
                    * self.get_a_s(self.x[i + 1], self.x[i])
                    * self.eps(T[i, j], self.rho(self.z[j]))
                    * 3
                    / (self.a * self.c)
                )

                a_p = -a_e - a_w - a_s - a_n

                p = self.m(i, j)
                b[p] = (
                    S
                    - a_p * T[i, j]
                    - a_w * T[im1, j]
                    - a_e * T[ip1, j]
                    - a_n_2
                    - a_s_2
                )


        return b

    # -------------------------------------------------------------------------------------------------- #

    # we set up to solve as a matrix system, with A*c = b where c is delta T**4, some small increase in T until we get a solution.
    def fill_A_dump(self, T, A):
        """ fill A coefficient matrix for solving system A*c = b """

        # what should Rin and Rout be for weighting for north and south points?? Total R of cylinder, or R of zone?
        for j in range(self.Nz):
            for i in range(self.Nx):  # i.e. i = 1:79
                p = self.m(i, j)
                if j == self.Nz - 1:
                    jp1 = j
                else:
                    jp1 = j + 1

                if i == 0:
                    a_w = 0.0
                else:
                    a_w = self.get_a_e(self.x[i], j) / (
                        (self.rhoz[jp1] + self.rhoz[j])
                        * 0.5
                        * self.get_kappa(
                            (T[i, j] + T[i, jp1]) * 0.5,
                            (self.rhoz[jp1] + self.rhoz[j]) * 0.5, j
                        )
                        * self.dx
                    )
                    A[p, self.m(i - 1, j)] = a_w

                if i == self.Nx - 1:
                    ip1 = self.Nx - 1
                    a_e = 0.0
                else:
                    ip1 = i + 1
                    a_e = self.get_a_e(self.x[i + 1], j) / (
                        (self.rhoz[jp1] + self.rhoz[j])
                        * 0.5
                        * self.get_kappa(
                            (T[i, j] + T[i, jp1]) * 0.5,
                            (self.rhoz[jp1] + self.rhoz[j]) * 0.5, j
                        )
                        * self.dx
                    )
                    A[p, self.m(i + 1, j)] = a_e
                if j == 0:
                    a_s = self.get_a_s(self.x[i+1], self.x[i])
                    # a_s = (
                    #     self.get_a_s(self.x[i+1], self.x[i])
                    #     / ((self.rhoz[j]+ self.rhoz[j])*0.5 * self.get_kappa((T[i,j]+self.Tcrust**4)*0.5,
                    #     (self.rhoz[j]+ self.rhoz[j])*0.5, j) * self.dz[j])
                    #  )
                    #a_s = 0.0
                    #a_s = self.Fbase * self.get_a_s(self.x[i + 1], self.x[i])
                    #a_s = self.get_a_s(self.x[i + 1], self.x[i])
                else:
                    a_s = self.get_a_s(self.x[i + 1], self.x[i]) / (
                        (self.rhoz[j - 1] + self.rhoz[j])
                        * 0.5
                        * self.get_kappa(
                            (T[i, j] + T[i, j - 1]) * 0.5,
                            (self.rhoz[j - 1] + self.rhoz[j]) * 0.5, j
                        )
                        * (self.dz[j - 1] + self.dz[j])
                        * 0.5
                    )
                    A[p, self.m(i, j - 1)] = a_s

                if j == self.Nz - 1:
                    hotspoty = np.arange(0, int(self.hsize / 2))
                    if self.bb_flag == 'bb' and i not in hotspoty:
                        a_n = (
                            self.sigma
                            * (T[i, j] + self.Ttop ** 4)
                            * 0.5
                            * self.get_a_s(self.x[i + 1], self.x[i])
                        )
                        # a_n = 0.0
                    else:
                        a_n = self.get_a_s(self.x[i + 1], self.x[i]) / (
                            self.rhoz[j]
                            * self.get_kappa((T[i, j]), self.rhoz[j], j)
                            * self.dz[j]
                        )

                else:
                    a_n = self.get_a_s(self.x[i + 1], self.x[i]) / (
                        (self.rhoz[jp1] + self.rhoz[j])
                        * 0.5
                        * self.get_kappa(
                            (T[i, j] + T[i, jp1]) * 0.5,
                            (self.rhoz[jp1] + self.rhoz[j]) * 0.5, j
                        )
                        * (self.dz[jp1] + self.dz[j])
                        * 0.5
                    )
                    A[p, self.m(i, j + 1)] = a_n

                a_p = -a_e - a_w - a_s - a_n

                A[p, p] = a_p

        return A

    # define function to fill b, the source term matrix:
    def fill_b_dump(self, T, b):
        """ fill b matrix for solving A*c = b """

        # Compute b
        for j in range(self.Nz):
            for i in range(self.Nx):  # i.e. i = 1:79
                if j == self.Nz - 1:
                    jp1 = j
                else:
                    jp1 = j + 1

                if i == 0:
                    im1 = 0
                    a_w = 0
                else:
                    im1 = i - 1
                    a_w = self.get_a_e(self.x[i], j) / (
                        (self.rhoz[jp1] + self.rhoz[j])
                        * 0.5
                        * self.get_kappa(
                            (T[i, j] + T[i, jp1]) * 0.5,
                            (self.rhoz[jp1] + self.rhoz[j]) * 0.5, j
                        )
                        * self.dx
                    )

                if i == self.Nx - 1:
                    ip1 = self.Nx - 1
                else:
                    ip1 = i + 1
                if j == 0:

                    #a_s = 0.0
                    a_s_2 = self.Fbase*self.get_a_s(self.x[i+1], self.x[i])
                    #a_s_2 = self.Fbase*self.get_a_s(self.x[i+1], self.x[i])

                    a_s = (
                        self.get_a_s(self.x[i+1], self.x[i])
                        / ((self.rhoz[j]+ self.rhoz[j])*0.5 * self.get_kappa((T[i,j]+self.Tcrust**4)*0.5,
                        (self.rhoz[j]+ self.rhoz[j])*0.5, j) * self.dz[j])
                     )
                    # a_s_2 = a_s * (T[i, j] + self.Tcrust ** 4) * 0.5
                    # a_s = 0.0
                    #a_s = self.get_a_s(self.x[i + 1], self.x[i])
                    # Fb = 1e19
                    #a_s_2 = self.Fbase * self.get_a_s(self.x[i + 1], self.x[i])

                # elif j == self.Nz -1:
                #     a_s = (
                #        self.get_a_s(self.x[i+1], self.x[i])
                #         / (self.rhoc[j] * self.get_kappa((T[i,j]),self.rhoc[j]) * self.dz[j])
                #     )
                #     a_s_2 = a_s * T[i, j - 1]
                else:
                    a_s = self.get_a_s(self.x[i + 1], self.x[i]) / (
                        (self.rhoz[j - 1] + self.rhoz[j])
                        * 0.5
                        * self.get_kappa(
                            (T[i, j] + T[i, j - 1]) * 0.5,
                            (self.rhoz[j - 1] + self.rhoz[j]) * 0.5, j
                        )
                        * (self.dz[j - 1] + self.dz[j])
                        * 0.5
                    )
                    a_s_2 = a_s * T[i, j - 1]

                if j == self.Nz - 1:
                    hotspoty = np.arange(0, int(self.hsize / 2))
                    if i in hotspoty:
                        if self.hsize == 0:
                            a_n = self.get_a_s(self.x[i + 1], self.x[i]) / (
                                self.rhoz[j]
                                * self.get_kappa(
                                    (T[i, j] + self.Ttop ** 4) * 0.5, self.rhoz[j], j
                                )
                                * self.dz[j]
                            )
                            a_n_2 = a_n * (T[i, j] + self.Ttop ** 4) * 0.5

                        else:
                            a_n_2 = a_n * self.Thotspot ** 4
                            a_n = self.get_a_s(self.x[i + 1], self.x[i]) / (
                                self.rhoz[j]
                                * self.get_kappa(
                                    (T[i, j] + self.Thotspot ** 4) * 0.5, self.rhoz[j], j
                                )
                                * self.dz[j]
                            )
                            a_n_2 = a_n * (T[i, j] + self.Thotspot ** 4) * 0.5

                    else:
                        if self.bb_flag == 'bb':
                            a_n = self.sigma * self.get_a_s(self.x[i + 1], self.x[i])
                            # a_n = 0.0
                            a_n_2 = (
                                self.sigma
                                * (T[i, j] + self.Ttop ** 4)
                                * 0.5
                                * self.get_a_s(self.x[i + 1], self.x[i])
                            )

                        else:
                            a_n = self.get_a_s(self.x[i + 1], self.x[i]) / (
                                self.rhoz[j]
                                * self.get_kappa(
                                    (T[i, j] + self.Ttop ** 4) * 0.5, self.rhoz[j], j
                                )
                                * self.dz[j]
                            )
                            a_n_2 = a_n * (T[i, j] + self.Ttop ** 4) * 0.5

                else:
                    a_n = self.get_a_s(self.x[i + 1], self.x[i]) / (
                        (self.rhoz[jp1] + self.rhoz[j])
                        * 0.5
                        * self.get_kappa(
                            (T[i, j] + T[i, jp1]) * 0.5,
                            (self.rhoz[jp1] + self.rhoz[j]) * 0.5, j
                        )
                        * (self.dz[jp1] + self.dz[j])
                        * 0.5
                    )
                    a_n_2 = a_n * T[i, j + 1]
                    #print(f'rho = {(self.rhoz[jp1] + self.rhoz[j]) * 0.5}')
                    #print(f'Tij  = {T[i, j]}, Tijp1 = {T[i, jp1]}')

                if j == self.Nz - 1:
                    jp1 = self.Nz - 1
                else:
                    jp1 = j + 1
                a_e = self.get_a_e(self.x[i + 1], j) / (
                    (self.rhoz[jp1] + self.rhoz[j])
                    * 0.5
                    * self.get_kappa(
                        (T[i, j] + T[i, jp1]) * 0.5,
                        (self.rhoz[jp1] + self.rhoz[j]) * 0.5, j
                    )
                    * self.dx
                )

                S = (
                    (self.rhoz[jp1] + self.rhoz[j])
                    * 0.5
                    * (self.get_h(j))
                    * self.get_a_s(self.x[i + 1], self.x[i])
                    * self.eps(
                        (T[i, j] + T[i, jp1]) * 0.5,
                        (self.rhoz[jp1] + self.rhoz[j]) * 0.5, j
                    )
                    * 3
                    / (self.a * self.c)
                )

                a_p = -a_e - a_w - a_s - a_n

                p = self.m(i, j)


                # if j < self.jsub + 1:
                #     b[p] = - a_p * T[i, j]
                #     - a_w * T[im1, j]
                #     - a_e * T[ip1, j]
                #     - a_n_2
                #     - a_s_2
                # else:
                b[p] = (
                    S
                    - a_p * T[i, j]
                    - a_w * T[im1, j]
                    - a_e * T[ip1, j]
                    - a_n_2
                    - a_s_2
                )
                #if j < self.jsub:
                    #print('in substrate:')
                    #print(f'j = {j}')
                    #print(f'S = {S}, a_p = {a_p}, a_w = {a_w}, a_s = {a_s_2}, a_e = {a_e}, a_n = {a_n_2}')

        return b

    def doint(self, T_initial):
        """function that solves the matrix equation A*c = b for c and solves for temperature """

        # now do time steps to find solution:
        tol = self.tol
        op = self.op
        maxIter = self.maxIter

        iteration = 0
        below_acc = False
        T_n = T_initial.copy()

        # Allocate memory for coefficient matrices:
        N = (self.Nx) * (self.Nz)
        A = np.zeros((N, N))
        b = np.zeros(N)

        print(" ------------- Solving for T, please wait -------------  ")
        while (iteration < self.maxIter) and (below_acc is False):
            print(f'integrating.. step {iteration}')
            # fill A and b matrices
            if self.rhoflag == 'dump':
                A = self.fill_A_dump(T_n, A)
                b = self.fill_b_dump(T_n, b)
            else:
                A = self.fill_A(T_n, A)
                b = self.fill_b(T_n, b)

            # solve matrix system A*c = b:
            # c = scipy.linalg.solve(A, b)
            c = scipy.sparse.linalg.spsolve(A, b)  # this is faster than linalg solve

            # Add c to T_n:
            below_acc = True
            for i in range(self.Nx):
                for j in range(self.Nz):
                    T_n[i, j] += self.op * c[self.m(i, j)]

                    if abs(c[self.m(i, j)] / T_n[i, j]) > self.tol:
                        below_acc = False

            iteration += 1
        print(f"Finished. Converged = {below_acc} in {iteration} steps")
        return T_n

    # -------------------------------------------------------------------------------------------------- #

    def Tign(self, y8, T, rho):
        """ takes in column depth and calculates ignition temperature, from Bildsten 1998 eqn 29 """
        val = []
        j = 100
        for i in range(len(y8)-1):
            val.append(1.83e8
            * self.get_kappa(np.power(T,4)[i], rho[i], j) ** (-1 / 10)
            * self.Y ** (-3 / 10)
            * self.mu ** (-1 / 5)
            * (self.g / 1e14) ** (-1 / 5)
            * y8[i] ** (-2 / 5))
        return (val
        )


    def get_ignition(self, ycol, T):
        """calculates ignition depth for temperature profiles and plots 10 y vs T profiles evenly spaced in the y direction with the ignition curve"""

        # define columns to plot:
        ncols = 10
        cols = np.arange(0, self.Nx, (self.Nx + 1) / ncols, dtype=int)
        cols2 = cols * self.dx

        # get column depths, temperatures and ignition temperatures for these columns
        Ts = {}
        Tign = {}

        for i in cols:
            Ts[f"{i}"] = T[i, :]
            Tign[f"{i}"] = self.Tign(ycol / 1e8)
        import color

        c = color.IsoColorBlind(ncols)
        # now plot
        fig, ax = plt.subplots(figsize=[10, 8])
        plt.title(
            plt.title(
                f"Contour of ignition conditions for Ths = {self.Thotspot:2.2E}, hsize = {self.hsize:2.2E}, density ="
                + self.rhoflag
            )
        )
        plt.plot(ycol, Tign[f"{i}"], ls="--", label="ignition", color="k")
        for i in range(len(cols)):
            plt.plot(
                ycol, Ts[f"{cols[i]}"], label=f"column {cols2[i]}/{self.Lx}", color=c[i]
            )

        # inset axis to zoom in on bottom:

        # inset axes....
        # axins = ax.inset_axes([0.5, 0.35, 0.25, 0.25])
        # for i in range(len(cols)):
        #     axins.plot(ycol, Ts[f"{cols[i]}"], color=c[i])
        # # sub region of the original image
        # x1, x2, y1, y2 = 2e7, 3.5e7, 4e7, 4.8e7
        # axins.set_xlim(x1, x2)
        # axins.set_ylim(y1, y2)
        # axins.set_xscale("log")
        # axins.set_yscale("log")
        # # axins.set_xticklabels('')
        # # axins.set_yticklabels('')

        # ax.indicate_inset_zoom(axins, label=None)
        hotspoty = np.arange(
            int((self.Nx) / 2 - self.hsize / 2), int((self.Nx + 1) / 2 + self.hsize / 2)
        )

        plt.text(
            5e3,
            3e10,
            f"hot spot is located at {hotspoty[0]*self.dx:.2f} - {hotspoty[-1]*self.dx:.2f}",
            color="grey",
        )

        plt.legend(loc="best")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("log column depth (g/cm^2)")
        plt.ylabel("log Temperature (K)")

        plt.savefig("plots/" + self.run_name + "_ignitioncurves.pdf")
        plt.close()

    # now solve and plot:
    def run(self):
        """ initialise mesh and solve for T """

        # save initial T values for plotting purposes later:
        T_initial = self.fill_Tinitial()

        with open("data/" + self.run_name + "_initialdata.txt", "wb") as f:
            np.savetxt(
                f,
                T_initial,
                fmt="%.18e",
                delimiter=" ",
                header="T[0,:], .. T[x,:]",
                comments="# everything is in cgs\n",
            )

        # plot initial conditions:
        # Set colour interpolation and colour map
        # cmap = color.ColorBlindRainbow()
        # colourMap = color.WaveFilter(cmap, nwaves = 40, amplitude=0.2)
        # can also try:
        # colorinterpolation = 50
        colourMap = plt.cm.magma  # you can try: colourMap = plt.cm.coolwarm

        # make meshgrid for contour plotting
        zplot, xplot = np.meshgrid(self.z, np.hstack((-self.x[1:][::-1], self.x)))
        T_p_in = np.vstack((T_initial[::-1, :], T_initial))

        # plot final result:
        f = plt.figure(figsize=(10, 5))
        plt.title(
            f"Contour of Temperature initial conditions for Ths = {self.Thotspot:2.2E}, hsize = {self.hsize:2.2E}, density ="
            + self.rhoflag
        )
        plt.pcolormesh(xplot, zplot, np.log10(np.power(T_p_in, 0.25)), cmap=colourMap)
        # plt.axis("equal")
        # Set Colorbar
        cbar = plt.colorbar()
        cbar.set_label('log(T)')
        plt.xlabel("x")
        plt.ylabel("z")
        # Save the result in the plot window
        plt.savefig("plots/" + self.run_name + "_initialT.pdf")
        plt.close()

        # plot kappa
        kappa = np.zeros((len(T_initial[:, 0]), len(T_initial[0, :])))
        colourMap = plt.cm.magma_r
        for i in range(0, len(T_initial[:, 0])):
            for j in range(0, len(T_initial[0, :])):
                if self.rhoflag == 'dump':
                    kappa[i, j] = self.get_kappa(T_initial[i, j], self.rhoc[j], j)
                else:
                    kappa[i, j] = self.get_kappa(T_initial[i, j], self.rho(self.z[j]), j)
        f = plt.figure(figsize=(10, 5))

        kappa_in = np.vstack((kappa[::-1, :], kappa))
        plt.pcolormesh(xplot, zplot, np.log10(kappa_in), cmap=colourMap)
        # plt.axis("equal")
        # Set Colorbar
        cbar = plt.colorbar()
        cbar.set_label('log(kappa)')
        plt.xlabel("x")
        plt.ylabel("z")
        # Save the result in the plot window
        plt.savefig("plots/" + self.run_name + "_initialkap.pdf")
        plt.close()

        with open("data/" + self.run_name + "_initialdata_kap.txt", "wb") as f:
            np.savetxt(
                f,
                kappa,
                fmt="%.18e",
                delimiter=" ",
                header="T[0,:], .. T[x,:]",
                comments="# everything is in cgs\n",
            )

        # -------------------------------------------------------------------------------------------------- #
        # get solution:
        T_n = self.doint(T_initial)
        print(T_n)

        # plot solution
        T_p = np.vstack((T_n[::-1, :], T_n))
        # plot final result:
        # Configure the contour
        norm = Normalize(
            vmin=np.log10(np.power(T_p, 0.25)).min(),
            vmax=np.log10(np.power(T_p, 0.25)).max(),
        )
        f = plt.figure(figsize=(10, 5))
        ax = f.add_subplot(111)
        colourMap = plt.cm.magma
        # cmap = color.ColorBlindRainbow()
        # colourMap = color.WaveFilter(cmap, nwaves = 40, amplitude=0.2)
        # ax.set_aspect('equal')
        plt.title(
            f"Contour of logTemperature for {self.coord} coordinates, Ths = {self.Thotspot:2.2E}, hsize = {self.hsize:2.2E}, density ="
            + self.rhoflag
        )
        cm = ax.pcolormesh(
            xplot, zplot, np.log10(np.power(T_p, 0.25)), cmap=colourMap
        )  # np.log10(
        # np.power(T_p, 0.25)),cmap=colourMap, norm=norm)
        # Set Colorbar
        cbar = f.colorbar(cm, label='log(T)', orientation='horizontal')
        plt.ylabel("z")
        plt.xlabel("x")
        # Show the result in the plot window
        plt.tight_layout()
        plt.savefig("plots/" + self.run_name + "_solutionlogT.pdf")
        plt.close()

        # plot solution with contours:
        colorinterpolation = 100
        colourMap = plt.cm.magma
        # colourMap = cmr.sunburst
        f = plt.figure(figsize=(10, 5))
        plt.title(
            f"Contour of logTemperature for Ths = {self.Thotspot:2.2E}, hsize = {self.hsize:2.2E}, density ="
            + self.rhoflag
        )
        plt.contourf(
            xplot[:-1, :-1],
            zplot[:-1, :-1],
            np.log10(np.power(T_p, 0.25)),
            levels=50,
            cmap=colourMap,
        )
        # plt.axis("equal")
        # Set Colorbar
        cbar = plt.colorbar()
        cbar.set_label('log(T)')
        plt.ylabel("z")
        plt.xlabel("x")
        # Show the result in the plot window
        plt.savefig("plots/" + self.run_name + "_solutionlogT_contour.pdf")
        plt.close()

        # plot kappa
        kappa = np.zeros((len(T_n[:, 0]), len(T_n[0, :])))
        colourMap = plt.cm.magma_r
        for i in range(0, len(T_n[:, 0])):
            for j in range(0, len(T_n[0, :])):

                kappa[i, j] = self.get_kappa(T_n[i, j], self.rhoc[j], j)

        with open("data/" + self.run_name + "_solutiondata_kap.txt", "wb") as f:
            np.savetxt(
                f,
                kappa,
                fmt="%.18e",
                delimiter=" ",
                header="T[0,:], .. T[x,:]",
                comments="# everything is in cgs\n",
            )

        kappa_p = np.vstack((kappa[::-1, :], kappa))

        f = plt.figure(figsize=(10, 5))
        ax = f.add_subplot(111)
        ax.set_aspect('equal')
        plt.title(
            f"Contour of logKappafor {self.coord} coordinates, Ths = {self.Thotspot:2.2E}, hsize = {self.hsize:2.2E}, density ="
            + self.rhoflag
        )
        cm = ax.pcolormesh(xplot, zplot, np.log10(kappa_p), cmap=colourMap)  # np.log10(
        # np.power(T_p, 0.25)),cmap=colourMap, norm=norm)
        # Set Colorbar
        cbar = f.colorbar(cm, label='log(Kappa)', orientation='horizontal')
        plt.ylabel("z")
        plt.xlabel("x")
        # Show the result in the plot window
        plt.tight_layout()
        plt.savefig("plots/" + self.run_name + "_solutionlogkappa.pdf")
        plt.close()

        # -------------------------------------------------------------------------------------------------- #
        # get column depth:

        f = plt.figure(figsize=(10, 8))
        if self.rhoflag == 'dump':
            rhovals = -1 * self.rhoz
            plt.plot(self.z, self.rhoz)
        else:
            rhovals = -1 * self.rho(self.z)
            plt.plot(self.z, self.rho(self.z))
        plt.xlabel('z')
        plt.ylabel('density')
        plt.savefig(f'density_{self.run_name}.pdf')
        ycol = integrate.cumtrapz(rhovals[::-1], self.z[::-1], initial=1e2)[::-1]

        # plot column depth:
        # Configure the contour
        yplot, xplot = np.meshgrid(ycol, np.hstack((-self.x[1:][::-1], self.x[1:])))

        # Set colour interpolation and colour map
        # cmap = color.ColorBlindRainbow()
        # colourMap = color.WaveFilter(cmap, nwaves=40, amplitude=0.2)
        colourMap = plt.cm.magma

        f = plt.figure(figsize=(10, 5))
        plt.title(
            f"Contour of logTemperature for {self.coord}, Ths = {self.Thotspot:2.2E}, hsize = {self.hsize:2.2E}, density ="
            + self.rhoflag
        )
        plt.pcolormesh(
            xplot, yplot, np.log10(np.power(T_p[1:, :], 0.25)), cmap=colourMap
        )
        # plt.axis("equal")
        # Set Colorbar
        cbar = plt.colorbar()
        cbar.set_label('log(T)')
        plt.gca().invert_yaxis()
        plt.yscale('log')
        plt.ylabel("log column depth")
        plt.xlabel("x")
        # Show the result in the plot window
        plt.savefig("plots/" + self.run_name + "_solutionlogT_columndepth.pdf")
        plt.close()

        # Calculate ignition conditions and plot temperature profiles for 10 evenly spaced columns:

        # self.get_ignition(ycol, np.power(T_n, 0.25))

        # finally, output all data to a file to save:
        # columns: z, ycol, Tx0..Txi

        with open("data/" + self.run_name + "_solutiondata.txt", "wb") as f:
            np.savetxt(
                f,
                T_n,
                fmt="%.18e",
                delimiter=" ",
                header="T[0,:], .. T[x,:]",
                comments="# everything is in cgs\n",
            )

        data2 = [
            self.Nx,
            self.Nz,
            self.Lx,
            self.Lz,
            self.Tcrust,
            self.Ttop,
            self.Thotspot,
            self.hsize,
            self.const_kappa,
            self.Fbase,
        ]
        # with open("data/" + self.run_name + "_initialdata.txt", "wb") as f:
        #     np.savetxt(
        #         f,
        #         data2,
        #         fmt="%.18e",
        #         delimiter=" ",
        #         header="Nx, Nz, Lx, Lz, Tcrust, Ttop, Thotspot, hsize, kappa, Fb",
        #         comments="# everything is in cgs, except hsize, Nx, Nz, which are in code grid size\n",
        #     )

    def read_file(self, run_name):
        # """ this file reads in a saved file from a run() instance """

        with open("data/" + run_name + "_initialdata.txt") as f:
            lines = (line for line in f if not line.startswith("#"))
            Nx, Nz, Lx, Lz, Tcrust, Ttop, Thotspot, hsize, kappa, Fbase = np.loadtxt(
                lines, delimiter=" ", skiprows=1
            )

        x = np.linspace(0, int(Lx), int(Nx) + 1)
        z = np.linspace(0, int(Lz), int(Nz) + 1)
        with open(run_name + "_solutiondata.txt") as f:
            lines = (line for line in f if not line.startswith("#"))
            T = np.loadtxt(lines, delimiter=" ", skiprows=1)

        return Nx, Nz, Lx, Lz, Tcrust, Ttop, Thotspot, hsize, kappa, Fbase, x, z, T

    def plot_from_save(self, run_name):
        """ option to make ignition and temperature contour plot from save file of previous run """
        import numpy as np
        import matplotlib.pyplot as plt
        import scipy.integrate as integrate

        (
            Nx,
            Nz,
            Lx,
            Lz,
            Tcrust,
            Ttop,
            Thotspot,
            hsize,
            kappa,
            Fbase,
            x,
            z,
            T_n,
        ) = self.read_file(run_name)
        # Set colour interpolation and colour map
        colorinterpolation = 50
        colourMap = plt.cm.jet  # you can try: colourMap = plt.cm.coolwarm
        # make meshgrid for contour plotting
        zplot, xplot = np.meshgrid(np.hstack(z[::-1], z), np.hstack(x[::-1], x))
        # plot final result:
        # Configure the contour
        f = plt.figure(figsize=(10, 5))
        plt.title("Contour of logTemperature Solution")
        plt.contourf(
            xplot,
            zplot,
            np.log10(np.hstack(np.power(T_n[::-1], 0.25), np.power(T_n, 0.25))),
            colorinterpolation,
            cmap=colourMap,
        )
        # Set Colorbar
        plt.colorbar()
        plt.ylabel("z")
        plt.xlabel("x")
        # Show the result in the plot window
        plt.savefig("data/" + run_name + "_solutionlogT.pdf")
        plt.close()

        # get column depth:
        rhovals = -1 * self.rho(z)
        ycol = integrate.cumtrapz(rhovals[::-1], z[::-1], initial=1e2)[::-1]

        # plot column depth:
        # Configure the contour
        yplot, xplot = np.meshgrid(ycol, x)

        f = plt.figure(figsize=(10, 5))
        plt.title("Contour of logTemperature Solution")
        plt.contourf(
            xplot,
            np.log10(yplot),
            np.power(T_n, 0.25),
            colorinterpolation,
            cmap=colourMap,
        )
        # Set Colorbar
        plt.colorbar()
        plt.gca().invert_yaxis()
        plt.ylabel("log y")
        plt.xlabel("x")
        # Show the result in the plot window
        plt.savefig(run_name + "_solutionlogT_columndepth.pdf")
        plt.close()

        # Calculate ignition conditions and plot temperature profiles for 10 evenly spaced columns:

        self.get_ignition(ycol, np.power(T_n, 0.25))
