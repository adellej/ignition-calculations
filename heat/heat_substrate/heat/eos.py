'''
EOS Table Module
'''

from pathlib import Path

import numpy as np
from scipy.interpolate import RectBivariateSpline
from matplotlib.colors import LogNorm, Normalize
from matplotlib import pylab as plt
from isotope import ion as I

import color
from kepdump import load as D

from .numeric import quad

class EOS(object):
    """
    Class to return p and u and their derivative w/r T or T^4
    """
    def __init__(self, filename = '~/python/heat/eos.npz'):
        data = np.load(Path(filename).expanduser(), allow_pickle=True)
        # apparently, np.load retains file descriptor in returned
        # object, so we do want to discard it by the end of the
        # __init__ routine.
        self.data = data['p'], data['u']
        self.tem = data['tem']
        self.den = data['den']
        self.abu = data['abu']
        print(f'using EOS table for {self.abu}')
        self.p_t4 = None
        self.u_t4 = None
        self.p = None
        self.u = None

    def __call__(self, T, rho, T4 = False, dT = True, drho=True):
        if T4:
            if self.p_t4 is None:
                p, u = self.data
                tem = quad(self.tem)
                den = self.den
                abu = self.abu
                self.p_t4 = RectBivariateSpline(tem, den, p, kx=3, ky=3, s=0)
                self.u_t4 = RectBivariateSpline(tem, den, u, kx=3, ky=3, s=0)
                self.limits_t4 = np.array([[min(tem), max(tem)], [min(den), max(den)]])
            f = self.p_t4
            g = self.u_t4
            limits = self.limits_t4
        else:
            if self.p is None:
                p, u = self.data
                tem = self.tem
                den = self.den
                abu = self.abu
                self.p = RectBivariateSpline(tem, den, p, kx=3, ky=3, s=0)
                self.u = RectBivariateSpline(tem, den, u, kx=3, ky=3, s=0)
                self.limits = np.array([[min(tem), max(tem)], [min(den), max(den)]])
            f = self.p
            g = self.u
            limits = self.limits
        T = np.maximum(limits[0,0], np.minimum(limits[0,1], T))
        rho = np.maximum(limits[1,0], np.minimum(limits[1,1], rho))
        p  = f(T, rho, dx=0, dy=0, grid=False)
        u  = g(T, rho, dx=0, dy=0, grid=False)
        result = [p, u]
        dp = []
        du = []
        if dT:
            pt = f(T, rho, dx=1, dy=0, grid=False)
            ut = g(T, rho, dx=1, dy=0, grid=False)
            dp.append(pt)
            du.append(ut)
        if drho:
            pd = f(T, rho, dx=0, dy=1, grid=False)
            ud = g(T, rho, dx=0, dy=1, grid=False)
            dp.append(pd)
            du.append(ud)
        result.extend(dp)
        result.extend(du)
        if len(result) == 0:
            return result[0]
        return result

    def plot(self, q = 'u', dump=None, heat=None, dump_sel=None, heat_sel=None, dump_label=None, heat_label=None):
        if q == 'p':
            data = np.log10(self.data[0])
            labelk = r'$\log\,P$ ($\mathrm{erg}\,\mathrm{cm}^{-3}$)'
        else:
            data = np.log10(self.data[1])
            labelk = r'$\log\,u$ ($\mathrm{erg}\,\mathrm{g}^{-1}$)'
        tem = np.log10(self.tem)
        den = np.log10(self.den)
        abu = self.abu
        fig = plt.figure()
        ax = fig.add_subplot()
        norm = Normalize
        vmin = np.min(data)
        vmax = np.max(data)
        vmin = 1.
        vmax = 30
        norm = norm(vmin=vmin, vmax=vmax)
        cmap = color.colormap('magma_r')
        ax.set_xlabel(r'$\log\,\rho$ ($\mathrm{g}\,\mathrm{cm}^{-3})$')
        ax.set_ylabel(r'$\log\,T$ ($\mathrm{K}$)')
        cm = ax.pcolormesh(den, tem, data,  norm=norm, cmap=cmap, shading='gouraud')
        cb = fig.colorbar(cm, label=labelk , orientation='vertical')
        comp = ', '.join(fr'${a*100:g}\,\%$ {I(i).mpl}' for i,a in abu[()].items())
        ax.text(0.01, 0.99, comp, color='w', transform=ax.transAxes, ha='left', va='top')

        legend = False
        if dump is not None:
            d = D(dump)
            if dump_sel is None:
                dump_sel = slice(d.jburnmin, -1)
            if dump_label is None:
                dump_label = 'Kepler'
            ax.plot(
                np.log10(d.dn[dump_sel]),
                np.log10(d.tn[dump_sel]),
                color = '#7fff7f',
                lw = 3,
                label = dump_label,
                )
            legend = True
        if heat is not None:
            if heat_label is None:
                heat_label = 'Heat'
            ax.plot(
                np.log10(heat.rhoc[heat_sel]),
                np.log10(heat.T[heat_sel]),
                color = '#7f7fff',
                lw = 3,
                label = heat_label,
                )
            legend = True
        if legend:
            ax.legend(
                loc = 'lower right',
                fontsize = 'small',
                edgecolor = 'none',
                facecolor = 'white',
                framealpha = 0.4,
                )
        fig.tight_layout()

class MultiEOS(object):
    def __init__(
            self,
            filenames = (
                '~/python/heat/eos.npz',
                '~/python/heat/eos_fe.npz',
                )
            ):
        self.multi = [eos(f) for f in filenames]
    def __call__(self, *args, **kwargs):
        """
        generic call function
        """
        raise NotImplementedError()

_eos = dict()

def eos(filename = '~/python/heat/eos.npz'):
    """
    factory function to avoid re-loading
    """
    if filename in _eos:
        return _eos[filename]
    if isinstance(filename, tuple):
        eos = MultiEOS(filename)
    else:
        eos = EOS(filename)
    _eos[filename] = eos
    return eos

'''
Examples:
from heat.eos import eos as E
e = E()
e.plot(q='p', dump='~/kepler/xrb/heat/heat2#240', dump_label=r'Model H2, $1.87\times10^8\,\mathrm{K}$',dump_sel=slice(51,-1))
'''
