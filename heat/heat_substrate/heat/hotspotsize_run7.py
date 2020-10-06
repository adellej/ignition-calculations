from heat3 import Diffuse


sizes = [0.0001, 0.001, 0.005, 0.01, 0.1, 0.3, 1.0, 1.5, 2.0]
lengths = [0.004, 0.04, 0.4, 1.2, 2.4, 4.0, 5.2, 6.4, 8.0]
names = [
    'run7_HS_size_1cm',
    'run7_HS_size_1m',
    'run7_HS_size_5m',
    'run7_HS_size_10m',
    'run7_HS_size_100m',
    'run7_HS_size_300m',
    'run7_HS_size_1km',
    'run7_HS_size_1.5km',
    'run7_HS_size_2km',
]
temps = [1e8, 5e7]

# fixed flux base boundary
# cartesian coordinates
# hotsot size always 1/4 of total domain
# Thot = 1e8
# dump 2 minute before burst


for T in temps:
    for i in range(len(names)):
        D = Diffuse(
            nx=400,
            nz=280,
            lx=sizes[i] * 10 * 1e5,
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
            xhot=sizes[i] * 1e5,
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
            kappa=(
                '/Users/adelle/Documents/ignition_calculations/cylindrical_coords/finite_volume_method/kappa.npz',
                'kappa_fe.npz',
            ),
            dump='/Users/adelle/Documents/ignition_calculations/kepler/run7/h#206',
            run_name=names[i],
            substrate=True,
        )
        D.plot()
