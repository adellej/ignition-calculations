# 1 minute: 206
# 40 sec: 208
# 30 sec: 210
# 20 sec: 214
# 10 sec: 217
# 0 sec: 229

from heat3 import Diffuse
import numpy as np

dump_list = ['h#202', 'h#203', 'h#204']  # , 'h#205']

names = ['run7_120sec', 'run7_107sec', 'run7_81sec']  # , 'run7_76sec']


# dump_list = [
#     'h#202',
#     # 'h#206',
#     # 'h#208',
#     # 'h#210',
#     # 'h#214',
#     # 'h#217',
#     # 'h#218',
#     # 'h#219',
#     # 'h#220',
#     # 'h#222',
#     # 'h#223',
#     # 'h#225',
#     # 'h#226',
#     # 'h#229',
# ]
# dump_list = ['h#214', 'h#215', 'h#216']

# temps = [3e7, 8e7, 1.7e8]
temps = [
    0,
    1e7,
    2e7,
    2.5e7,
    3e7,
    4e7,
    5e7,
    6e7,
    8e7,
    9e7,
    1e8,
    1.05e8,
    1.1e8,
    1.2e8,
    1.4e8,
    1.6e8,
    1.7e8,
    1.8e8,
    2e8,
]
# names = ['run7_20sec', 'run7_14sec', 'run7_13sec']
# names = ['run7_40sec','run7_30sec', 'run7_20sec','run7_10sec', 'run7_7sec', 'run7_6sec','run7_5sec','run7_4sec','run7_3sec','run7_2sec','run7_1sec','run7_0sec']
# names = [
#     'run7_120sec',
#     # 'run7_60sec',
#     # 'run7_40sec',
#     # 'run7_30sec',
#     # 'run7_20sec',
#     # 'run7_10sec',
#     # 'run7_6sec',
#     # 'run7_5sec',
#     # 'run7_4sec',
#     # 'run7_3sec',
#     # 'run7_2sec',
#     # 'run7_1sec',
#     # 'run7_0sec',
# ]


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
                geo='cartesian',
                mbase=3,
                mtop=5,  # BLACKBODY | TEMPERATURE
                surf='r',
                tbase=1e8,
                fbase=1e19,
                ttop=1e6,
                thot=temps[j],
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
                dump='/Users/adelle/Documents/ignition_calculations/kepler/run7/'
                + dump_list[i],
                run_name=names[i],
                substrate=True,
            )
            D.plot()

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
                geo='cartesian',
                mbase=3,
                mtop=5,  # BLACKBODY | TEMPERATURE
                surf='r',
                tbase=1e8,
                fbase=1e19,
                ttop=1e6,
                thot=temps[j],
                xhot=100000,
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
                dump='/Users/adelle/Documents/ignition_calculations/kepler/run7/'
                + dump_list[i],
                run_name=names[i],
                substrate=True,
            )
            D.plot()
