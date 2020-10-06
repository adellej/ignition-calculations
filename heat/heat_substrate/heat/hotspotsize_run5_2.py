from heat3 import Diffuse
import numpy as np

sizes = 10 ** (np.arange(61) * 0.1)
names = []
for i in range(len(sizes)):
    names.append(f'run5_HS_size_{sizes[i]}')
# names = ['run5_HS_size_0']
temps = [8e7]

for T in temps:
    for i in range(len(names)):

        D = Diffuse(
            nx=500,
            nz=280,
            lx=sizes[i] * 10,
            ly=1,
            lz=600,
            x0=0,
            y0=0,
            z0=0,
            r=1e6,
            geo='cylindrical',
            mbase=3,
            mtop=5,  # BLACKBODY | TEMPERATURE
            surf='r',
            tbase=1e8,
            fbase=1e19,
            ttop=1e6,
            thot=T,
            xhot=sizes[i],
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
            Y=0.99,
            net='static',
            bc=None,
            logfac=0.1,
            kappa=(
                '/Users/adelle/Documents/ignition_calculations/cylindrical_coords/finite_volume_method/kappa.npz',
                'kappa_fe.npz',
            ),
            dump='/Users/adelle/Documents/ignition_calculations/kepler/run5/h#945',
            run_name='substrate' + names[i],
            substrate=True,
        )
        D.plot()


sizes = 10 ** (np.arange(61) * 0.1)
names = []
for i in range(len(sizes)):
    names.append(f'run5_HS_size_{sizes[i]}_cart')

temps = [8e7]

for T in temps:
    for i in range(len(names)):

        D = Diffuse(
            nx=500,
            nz=280,
            lx=sizes[i] * 10,
            ly=1,
            lz=600,
            x0=0,
            y0=0,
            z0=0,
            r=1e6,
            geo='cartesian',
            mbase=3,
            mtop=5,  # BLACKBODY | TEMPERATURE
            surf='r',
            tbase=1e8,
            fbase=1e19,
            ttop=1e6,
            thot=T,
            xhot=sizes[i],
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
            Y=0.99,
            net='static',
            bc=None,
            logfac=0.1,
            kappa=(
                '/Users/adelle/Documents/ignition_calculations/cylindrical_coords/finite_volume_method/kappa.npz',
                'kappa_fe.npz',
            ),
            dump='/Users/adelle/Documents/ignition_calculations/kepler/run5/h#945',
            run_name='substrate' + names[i],
            substrate=True,
        )
        D.plot()

