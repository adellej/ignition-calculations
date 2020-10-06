from heat3 import Diffuse
import numpy as np


# dump_list = ['h#298']
dump_list = ['h#295']
names = ['run3']
# temps = [2e7]
temps = [2e7, 8e7, 1e8, 1.1e8, 1.5e8]
# size = 300000

# dump_list = ['h#946']
# names = ['run5']
# temps = [1.1e8]  # [2e7, 8e7, 1e8, 1.1e8]
# size = 50000

# dump_list = ['h#206']
# names = ['run7']
# temps = [1e7, 2e7, 8e7, 9e7, 1e8]
# temps = [2e7]
# size = 100000

# with open(f"data/run5/data_solutiontemp_runsubstraterun5_qb1_hs100000000.0.txt") as f:
#     lines = (line for line in f if not line.startswith("#"))
#     oldT = np.loadtxt(lines, delimiter=" ", skiprows=1)

for j in range(len(temps)):
    D = Diffuse(
        nx=200,
        nz=280,
        lx=4e5,
        ly=1,
        lz=600,
        x0=0,
        y0=0,
        z0=0,
        r=1e6,
        geo='cylindrical',
        mbase=2,
        mtop=5,  # BLACKBODY | TEMPERATURE
        surf='r',
        tbase=1e8,
        fbase=1e19,
        ttop=1e6,
        thot=temps[j],
        xhot=3e5,
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
        ),  # <filename
        dump=f'/Users/adelle/Documents/ignition_calculations/kepler/{names[0]}/'
        + dump_list[0],
        run_name=names[0],
        substrate=True,
    )
    D.plot()
