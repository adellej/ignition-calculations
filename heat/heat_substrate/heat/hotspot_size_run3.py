from heat3 import Diffuse
import numpy as np

# sizes = [0, 1e-2, 1e-1, 1]
# lengths = 10 * sizes
sizes = 10 ** (np.arange(61) * 0.1)
# lengths = [0.001, 0.01, 0.1, 1.0, 10.0, 20.0]

# names = [
#     'run3_HS_size_1cm_cart',
#     'run3_HS_size_1m_cart',
#     'run3_HS_size_5m_cart',
#     'run3_HS_size_10m_cart',
#     'run3_HS_size_100m_cart',
#     'run3_HS_size_300m_cart',
#     'run3_HS_size_1km_cart',
#     'run3_HS_size_1.5km_cart',
#     'run3_HS_size_2km_cart',
# ]
# names = []
# for i in range(len(sizes)):
#     names.append(f'run3_HS_size_{sizes[i]}')
# names = [
#     'run3_HS_size_0cm',
#     'run3_HS_size_0.01cm',
#     'run3_HS_size_0.1cm',
#     'run3_HS_size_1cm',
# ]
names = ['run3_HS_size_0']

# temps = [0, 3e7, 4e7, 5e7, 6e7, 8e7, 9e7, 1e8, 1.7e8, 2e8]
temps = [8e7]
# fixed flux base boundary
# cylindrical coordinates
# hotsot size always 1/4 of total domain
# Thot = 1e8
# dump 1 minute before burst
# 170 177 minutes before burst

for T in temps:
    for i in range(len(names)):

        D = Diffuse(
            nx=500,
            nz=280,
            # lx=sizes[i] * 10,
            lx=1,
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
            # xhot=sizes[i],
            xhot=0,
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
            dump='/Users/adelle/Documents/ignition_calculations/kepler/run3/h#292',
            run_name='substrate' + names[i],
            substrate=True,
        )
        D.plot()
