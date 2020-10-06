from heat3 import Diffuse
import numpy as np

# run3:
# 1 minute: 298
# 40 sec: 301
# 30 sec: 303
# 20 sec: 306
# 10 sec: 309
# 0 sec: 321


# dump_list = ['h#301', 'h#303', 'h#306', 'h#309', 'h#321']
dump_list = [
    'h#298',
    'h#301',
    'h#303',
    'h#306',
    'h#309',
    'h#310',
    'h#311',
    'h#312',
    'h#314',
    'h#315',
    'h#317',
    'h#321',
]
# dump_list = ['h#307', 'h#308']
# 5 sec: 311
# 4 sec: 312
# 3 sec: 314
# 2 sec: 315
# 1 sec: 317
# 0 sec: 321
temps2 = [
    # 0,
    # 3e7,
    # 4e7,
    # 5e7,
    # 6e7,
    # 8e7,
    # 9e7,
    # 1e8,
    # 1.05e8,
    # 1.1e8,
    # 1.12e8,
    1.14e8,
    1.16e8,
    1.18e8,
    1.2e8,
    1.4e8,
    1.6e8,
    1.7e8,
    1.8e8,
    2e8,
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
    1.1e8,
    1.12e8,
    1.14e8,
    1.16e8,
    1.18e8,
    1.2e8,
    1.4e8,
    1.6e8,
    1.7e8,
    1.8e8,
    2e8,
]
# temps = [1.7e8]
# names = ['run3_40sec','run3_30sec', 'run3_20sec','run3_10sec', 'run3_0sec']
names = [
    'run3_60sec',
    'run3_40sec',
    'run3_30sec',
    'run3_20sec',
    'run3_10sec',
    #'run3_6sec',
    # 'run3_5sec',
    # 'run3_4sec',
    # 'run3_3sec',
    # 'run3_2sec',
    # 'run3_1sec',
    # 'run3_0sec',
]
# names = ['run3_14sec', 'run3_13sec']
for j in range(len(temps)):
    for i in range(len(names)):
        # if j == 0:
        #     D = Diffuse(
        #         nx=200,
        #         nz=280,
        #         lx=400000,
        #         ly=1,
        #         lz=600,
        #         x0=0,
        #         y0=0,
        #         z0=0,
        #         r=1e6,
        #         geo='cylindrical',
        #         mbase=3,
        #         mtop=5,  # BLACKBODY | TEMPERATURE
        #         surf='r',
        #         tbase=1e8,
        #         fbase=1e19,
        #         ttop=1e6,
        #         thot=temps[j],
        #         xhot=100000,
        #         tol=1e-6,
        #         op=None,
        #         rhoflag='fit',
        #         method='solve4',
        #         normalize=False,
        #         T=None,
        #         sparse=True,
        #         initial=False,
        #         jac=True,
        #         xres='constant',  # constant | progressive
        #         yres='constant',  # ...
        #         zres='constant',  # ... | logarithmic | surflog
        #         root_options=None,
        #         root4=True,
        #         Y=0.99,
        #         net='static',
        #         bc=None,
        #         logfac=0.1,
        #         kappa=(
        #             '/Users/adelle/Documents/ignition_calculations/cylindrical_coords/finite_volume_method/kappa.npz',
        #             'kappa_fe.npz',
        #         ),  # <filename
        #         dump='/Users/adelle/Documents/ignition_calculations/kepler/run3/'
        #         + dump_list[i],
        #         run_name=names[i],
        #         substrate=True,
        #     )
        #     D.plot()

        # else:

        if j == 0:

            with open(
                f"data/data_solutiontemp_runsubstrate{names[i]}_qb{1}_hs{temps2[0]}.txt"
            ) as f:
                lines = (line for line in f if not line.startswith("#"))
                oldT = np.loadtxt(lines, delimiter=" ", skiprows=1)

        else:

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
            xhot=300000,
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
            ),  # <filename
            dump='/Users/adelle/Documents/ignition_calculations/kepler/run3/'
            + dump_list[i],
            run_name=names[i],
            substrate=True,
        )
        D.plot()
