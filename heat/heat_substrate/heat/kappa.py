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

    def __init__(self, filename='~/python/heat/kappa.npz'):
        data = np.load(Path(filename).expanduser(), allow_pickle=True)
        # apparently, np.load retains file descriptor in returned
        # object, so we do want to discard it by the end of the
        # __init__ routine.
        self.data = data['kappa']
        self.tem = data['tem']
        self.den = data['den']
        self.abu = data['abu']
        print(f'using opacity table for {self.abu}')
        self.kappai_t4 = None
        self.kappai = None

    def __call__(self, T, rho, /, T4=False, dT=True, drho=True):
        if T4:
            if self.kappai_t4 is None:
                kappai = 1 / self.data
                tem = quad(self.tem)
                den = self.den
                abu = self.abu
                self.kappai_t4 = RectBivariateSpline(tem, den, kappai, kx=3, ky=3, s=0)
                self.kappa_limits_t4 = np.array(
                    [[min(tem), max(tem)], [min(den), max(den)]]
                )
            f = self.kappai_t4
            limits = self.kappa_limits_t4
        else:
            if self.kappai is None:
                kappai = 1 / self.data
                tem = self.tem
                den = self.den
                abu = self.abu
                self.kappai = RectBivariateSpline(tem, den, kappai, kx=3, ky=3, s=0)
                self.kappa_limits = np.array(
                    [[min(tem), max(tem)], [min(den), max(den)]]
                )
            f = self.kappai
            limits = self.kappa_limits
        T = np.maximum(limits[0, 0], np.minimum(limits[0, 1], T))
        rho = np.maximum(limits[1, 0], np.minimum(limits[1, 1], rho))
        kappai = f(T, rho, dx=0, dy=0, grid=False)
        result = [kappai]
        if dT:
            kappait = 0.5 * f(T, rho, dx=1, dy=0, grid=False) / kappai
            result.append(kappait)
        if drho:
            kappaid = 0.5 * f(T, rho, dx=0, dy=1, grid=False) / kappai
            result.append(kappaid)
        if len(result) == 0:
            return result[0]
        return result

    def plot(
        self,
        dump=None,
        heat=None,
        dump_sel=None,
        heat_sel=None,
        dump_label=None,
        heat_label=None,
    ):
        kappa = np.log10(self.data)
        tem = np.log10(self.tem)
        den = np.log10(self.den)
        abu = self.abu
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
        cm = ax.pcolormesh(den, tem, kappa, norm=norm, cmap=cmap, shading='gouraud')
        cb = fig.colorbar(cm, label=labelk, orientation='vertical')
        comp = ', '.join(fr'${a*100:g}\,\%$ {I(i).mpl}' for i, a in abu[()].items())
        ax.text(
            0.01, 0.99, comp, color='w', transform=ax.transAxes, ha='left', va='top'
        )

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
                color='#7fff7f',
                lw=3,
                label=dump_label,
            )
            legend = True
        if heat is not None:
            if heat_label is None:
                heat_label = 'Heat'
            ax.plot(
                np.log10(heat.rhoc[heat_sel]),
                np.log10(heat.T[heat_sel]),
                color='#7f7fff',
                lw=3,
                label=heat_label,
            )
            legend = True
        if legend:
            ax.legend(
                loc='lower right',
                fontsize='small',
                edgecolor='none',
                facecolor='white',
                framealpha=0.4,
            )
        fig.tight_layout()


class Multi(object):
    def __call__(self, *args, **kwargs):
        """
        generic call function
        """
        table = kwargs.pop('table', 0)
        T = args[0]
        rho = args[1]
        dT = kwargs.get('dT', True)
        drho = kwargs.get('drho', True)
        if np.ndim(table) == 0:
            table = np.full(T.shape, table)
        assert T.shape == table.shape == rho.shape
        data = np.full(T.shape, np.nan)
        nret = 1
        if dT:
            data_dT = data.copy()
            nret += 1
        if drho:
            data_drho = data.copy()
            nret += 1
        for i in np.unique(table):
            ii = table == i
            x = self.multi[i](T[ii], rho[ii], **kwargs)
            if nret == 1:
                data[ii] = x
            else:
                data[ii] = x[0]
                iret = 1
                if dT:
                    data_dT[ii] = x[iret]
                    iret += 1
                if drho:
                    data_drho[ii] = x[iret]
        if nret == 1:
            return data
        retval = [data]
        if dT:
            retval.append(data_dT)
        if drho:
            retval.append(data_drho)
        return tuple(retval)


class MultiKappa(Multi):
    def __init__(
        self, filenames=('~/python/heat/kappa.npz', '~/python/heat/kappa_fe.npz',)
    ):
        self.multi = [kappa(f) for f in filenames]


_kappa = dict()


def kappa(filename='~/python/heat/kappa.npz'):
    """
    factory function to avoid re-loading
    """
    if filename in _kappa:
        return _kappa[filename]
    if isinstance(filename, tuple):
        k = MultiKappa(filename)
    else:
        k = Kappa(filename)
    _kappa[filename] = k
    return k


'''
Examples:
from heat.kappa import kappa as K
k = K()
k.plot(dump='~/kepler/xrb/heat/heat2#240', dump_label=r'Model H2, $2.01\times10^8\,\mathrm{K}$',dump_sel=slice(51,-1))
k = K('~/python/heat/kappa_fe.npz')
k.plot(dump='~/kepler/xrb/heat/heat2#240', dump_label=r'Model H2, $5.04\times10^8\,\mathrm{K}$',dump_sel=slice(1,51))
'''
