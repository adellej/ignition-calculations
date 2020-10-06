from heat3 import Diffuse
import numpy as np

# run5:
# 1 minute: 946
# 40 sec: 947
# 30 sec: 948
# 20 sec: 949
# 10 sec: 952
# 0 sec: 966
# dump_list = ['h#946', 'h#947', 'h#948', 'h#949', 'h#950', 'h#952']  # , 'h#966']
dump_list = ['h#953', 'h#954', 'h#955', 'h#957', 'h#958']  # , 'h#960']
# 1 sec 960
# 2 sec 958
# 3 sec 957
# 4 sec
# 5 sec 955
# 7 sec 954
# 8 sec 953

# temps = [3e7, 8e7, 1.7e8]
temps2 = [
    0,
    # 3e7,
    # 4e7,
    # 5e7,
    # 6e7,
    # 8e7,
    # 9e7,
    # 1e8,
    # 1.05e8,
    # 1.055e8,
    # 1.06e8,
    # 1.065e8,
    # 1.07e8,
    # 1.09e8,
    # 1.1e8,
    # 1.2e8,
    # 1.4e8,
    # 1.6e8,
    # 1.7e8,
    # 1.8e8,
    # 2e8,
]
temps = [
    0,
    3e7,
    4e7,
    5e7,
    6e7,
    8e7,
    9e7,
    1e8,
    1.05e8,
    1.055e8,
    1.06e8,
    1.065e8,
    1.07e8,
    1.09e8,
    1.1e8,
    1.2e8,
    1.4e8,
    1.6e8,
    1.7e8,
    1.8e8,
    2e8,
]
# names = [
#     'run5_60sec',
#     'run5_40sec',
#     'run5_30sec',
#     'run5_20sec',
#     'run5_16sec',
#     'run5_10sec',
#     #'run5_0sec',
# ]
# 1 sec 960
# 2 sec 958
# 3 sec 957
# 4 sec
# 5 sec 955
# 7 sec 954
# 8 sec 953
names = [
    'run5_8sec',
    'run5_7sec',
    'run5_5sec',
    'run5_3sec',
    'run5_2sec',
]  # , 'run5_1sec']


for j in range(len(temps)):
    for i in range(len(names)):

        if j == 0:

            D = Diffuse(
                nx=200,
                nz=280,
                lx=400000,
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
                thot=temps[j],
                xhot=50000,
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
                dump='/Users/adelle/Documents/ignition_calculations/kepler/run5/'
                + dump_list[i],
                run_name=names[i],
                substrate=True,
            )

            D.plot()

        else:

            # if j == 0:

            #     with open(
            #         f"data/data_solutiontemp_runsubstrate{names[i]}_qb{1}_hs{temps2[0]}.txt"
            #     ) as f:
            #         lines = (line for line in f if not line.startswith("#"))
            #         oldT = np.loadtxt(lines, delimiter=" ", skiprows=1)

            # else:

            with open(
                f"data/data_solutiontemp_runsubstrate{names[i]}_qb{1}_hs{temps[j-1]}.txt"
            ) as f:
                lines = (line for line in f if not line.startswith("#"))
                oldT = np.loadtxt(lines, delimiter=" ", skiprows=1)

            D = Diffuse(
                nx=200,
                nz=280,
                lx=400000,
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
                thot=temps[j],
                xhot=50000,
                tol=1e-6,
                op=None,
                rhoflag='fit',
                method='solve4',
                normalize=False,
                T=oldT,
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
                dump='/Users/adelle/Documents/ignition_calculations/kepler/run5/'
                + dump_list[i],
                run_name=names[i],
                substrate=True,
            )

            D.plot()

