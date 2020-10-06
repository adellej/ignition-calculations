'''
c_P Table Module
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

from .kappa import Multi

class Cp(object):
    """
    Class to return cp and and . derivative w/r T or T^4
    """
    def __init__(self, filename = '~/python/heat/cp.npz'):
        data = np.load(Path(filename).expanduser(), allow_pickle=True)
        # apparently, np.load retains file descriptor in returned
        # object, so we do want to discard it by the end of the
        # __init__ routine.
        self.data = data['cp']
        self.tem = data['tem']
        self.den = data['den']
        self.abu = data['abu']
        print(f'using cp table for {self.abu}')
        self.cp_t4 = None
        self.cp = None

    def __call__(self, T, rho, T4 = True, dT = True, drho=True):
        if T4:
            if self.cp_t4 is None:
                cp = self.data
                tem = quad(self.tem)
                den = self.den
                abu = self.abu
                self.cp_t4 = RectBivariateSpline(tem, den, cp, kx=3, ky=3, s=0)
                self.cp_limits_t4 = np.array([[min(tem), max(tem)], [min(den), max(den)]])
            f = self.cp_t4
            limits = self.cp_limits_t4
        else:
            if self.cp is None:
                cp = self.data
                tem = self.tem
                den = self.den
                abu = self.abu
                self.cp = RectBivariateSpline(tem, den, cp, kx=3, ky=3, s=0)
                self.cp_limits = np.array([[min(tem), max(tem)], [min(den), max(den)]])
            f = self.cp
            limits = self.cp_limits
        T = np.maximum(limits[0,0], np.minimum(limits[0,1], T))
        rho = np.maximum(limits[1,0], np.minimum(limits[1,1], rho))
        cp  = f(T, rho, dx=0, dy=0, grid=False)
        result = [cp]
        if dT:
            cpt = f(T, rho, dx=1, dy=0, grid=False)
            result.append(cpt)
        if drho:
            cpd = f(T, rho, dx=0, dy=1, grid=False)
            result.append(cpd)
        if len(result) == 0:
            return result[0]
        return result

    def plot(self, dump=None, heat=None, dump_sel=None, heat_sel=None, dump_label=None, heat_label=None):
        cp = np.log10(self.data)
        # cp = self.data
        tem = np.log10(self.tem)
        den = np.log10(self.den)
        abu = self.abu
        fig = plt.figure()
        ax = fig.add_subplot()
        norm = Normalize
        vmin = np.min(cp)
        vmax = np.max(cp)
        vmin = 1.
        vmax = 30
        norm = norm(vmin=vmin, vmax=vmax)
        cmap = color.colormap('magma_r')
        ax.set_xlabel(r'$\log\,\rho$ ($\mathrm{g}\,\mathrm{cm}^{-3})$')
        ax.set_ylabel(r'$\log\,T$ ($\mathrm{K}$)')
        labelk = r'$\log\,c_{\mathrm{p}}$ ($\mathrm{erg}\,\mathrm{g}^{-1}\,\mathrm{K}^{-1}$)'
        cm = ax.pcolormesh(den, tem, cp,  norm=norm, cmap=cmap, shading='gouraud')
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

class MultiCp(Multi):
    def __init__(
            self,
            filenames = (
                '~/python/heat/cp.npz',
                '~/python/heat/cp_fe.npz',
                )
            ):
        self.multi = [cp(f) for f in filenames]

_cp = dict()

def cp(filename = '~/python/heat/cp.npz'):
    """
    factory function to avoid re-loading
    """
    if filename in _cp:
        return _cp[filename]
    if isinstance(filename, tuple):
        c = MultiCp(filename)
    else:
        c = Cp(filename)
    _cp[filename] = c
    return c

'''
Examples:
from heat.cp import cp as C
c = C()
c.plot(dump='~/kepler/xrb/heat/heat2#240', label=r'Model H2, $1.87\times10^8\,\mathrm{K}$',dump_sel=slice(51,-1))
'''
