'''
Opacity Table Module
'''

from pathlib import Path

import numpy as np
from scipy.interpolate import RectBivariateSpline
from matplotlib.colors import LogNorm, Normalize
from matplotlib import pylab as plt
from isotope import ion as I

import color
from kepdump import load as D

from numeric import quad

class Kappa(object):
    """
    Class to return 1/kappa and and (1/2) log . derivative w/r T or T^4
    """
    def __init__(self, filename = '~/python/heat/kappa.npz'):
        self.kappa_data = np.load(Path(filename).expanduser(), allow_pickle=True)
        print(f'using opacity table for {self.kappa_data["abu"]}')
        self.kappai_t4 = None
        self.kappai = None

    def __call__(self, T, rho, T4 = True, dT = True):
        if T4:
            if self.kappai_t4 is None:
                data = self.kappa_data
                kappa = data['kappa']
                tem = quad(data['tem'])
                den = data['den']
                abu = data['abu']
                self.kappai_t4 = RectBivariateSpline(tem, den, 1/kappa, kx=2, ky=2, s=1)
                self.kappa_limits_t4 = np.array([[min(tem), max(tem)], [min(den), max(den)]])
            f = self.kappai_t4
            limits = self.kappa_limits_t4
        else:
            if self.kappai is None:
                data = self.kappa_data
                kappa = data['kappa']
                tem = data['tem']
                den = data['den']
                abu = data['abu']
                self.kappai = RectBivariateSpline(tem, den, 1/kappa, kx=2, ky=2, s=1)
                self.kappa_limits = np.array([[min(tem), max(tem)], [min(den), max(den)]])
            f = self.kappai
            limits = self.kappa_limits
        T = np.maximum(limits[0,0], np.minimum(limits[0,1], T))
        rho = np.maximum(limits[1,0], np.minimum(limits[1,1], rho))
        kappai  = f(T, rho, dx=0, dy=0, grid=False)
        if not dT:
            return kappai
        kappait = 0.5 * f(T, rho, dx=1, dy=0, grid=False) / kappai
        return kappai, kappait

    def plot(self, dump=None, sel=None, label=None):
        data = self.kappa_data
        kappa = np.log10(data['kappa'])
        tem = np.log10(data['tem'])
        den = np.log10(data['den'])
        abu = data['abu']
        fig = plt.figure()
        ax = fig.add_subplot()
        norm = Normalize
        vmin = np.min(kappa)
        vmax = np.max(kappa)
        norm = norm(vmin=vmin, vmax=vmax)
        cmap = color.colormap('magma_r')
        ax.set_xlabel(r'$\log\,\rho$ ($\mathrm{g}\,\mathrm{cm}^{-3})$')
        ax.set_ylabel(r'$\log\,T$ ($\mathrm{K}$)')
        labelk = r'$\log\,\kappa$ ($\mathrm{cm}^2\,\mathrm{g}^{-1}$)'
        cm = ax.pcolormesh(den, tem, kappa,  norm=norm, cmap=cmap, shading='gouraud')
        cb = fig.colorbar(cm, label=labelk , orientation='vertical')
        comp = ', '.join(fr'${a*100:g}\,\%$ {I(i).mpl}' for i,a in abu[()].items())
        ax.text(0.01, 0.99, comp, color='w', transform=ax.transAxes, ha='left', va='top')

        if dump is not None:
            d = D(dump)
            if sel is None:
                sel = slice(d.jbmasslow, -1)
            if label is None:
                label = 'Model'
            ax.plot(
                np.log10(d.dn[sel]),
                np.log10(d.tn[sel]),
                color = '#7fff7f',
                lw = 3,
                label = label,
                )
            ax.legend(
                loc = 'lower right',
                fontsize = 'small',
                edgecolor = 'none',
                facecolor = 'white',
                framealpha = 0.4,
                )
        fig.tight_layout()

_kappa = dict()

def kappa(filename = '~/python/heat/kappa.npz'):
    """
    factory function to avoid re-loading
    """
    if filename in _kappa:
        return _kappa[filename]
    k = Kappa(filename)
    _kappa[filename] = k
    return k

'''
Examples:
from heat.kappa import kappa as K
k = K()
k.plot(dump='~/kepler/xrb/heat/heat2#240', label=r'Model H2, $1.87\times10^8\,\mathrm{K}$',sel=slice(51,-1))
'''
