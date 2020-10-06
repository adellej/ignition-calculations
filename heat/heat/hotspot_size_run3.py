from heat3 import Diffuse

sizes = [0.0001,0.001, 0.005, 0.01,0.1, 0.3, 1.0, 1.5, 2.0]
lengths = [0.004, 0.04, 0.4, 1.2, 2.4, 4.0, 5.2, 6.4, 8.0]
names = ['run3_HS_size_1cm_cart','run3_HS_size_1m_cart', 'run3_HS_size_5m_cart', 'run3_HS_size_10m_cart', 'run3_HS_size_100m_cart', 'run3_HS_size_300m_cart', 'run3_HS_size_1km_cart', 'run3_HS_size_1.5km_cart', 'run3_HS_size_2km_cart']
temps =[1e8, 5e7]
# fixed flux base boundary
# cylindrical coordinates
# hotsot size always 1/4 of total domain
# Thot = 1e8
# dump 1 minute before burst
# 170 177 minutes before burst

for T in temps:
    for i in range(len(names)):
            D = Diffuse(
                nx=400,
                nz = 280,
                lx = sizes[i]*10*1e5,
                ly = 1,
                lz = 600,
                x0 = 0,
                y0 = 0,
                z0 = 0,
                r = 1e6,
                geo = 'cartesian',
                mbase = 2,
                mtop = 5, # BLACKBODY | TEMPERATURE
                surf = 'r',
                tbase = 1e8,
                fbase = 1e19,
                ttop = 1e6,
                thot = T,
                xhot = sizes[i]*1e5,
                #xhot = 0,
                tol = 1e-6,
                op = None,
                rhoflag = 'fit',
                method = 'solve4',
                normalize = False,
                T = None,
                sparse = True,
                initial = False,
                jac = True,
                xres = 'constant', # constant | progressive
                yres = 'constant', # ...
                zres = 'constant', # ... | logarithmic | surflog
                root_options = None,
                root4 = True,
                Y = 1,
                net = 'static',
                bc = None,
                logfac = 0.1,
                kappa = '/Users/adelle/Documents/ignition_calculations/cylindrical_coords/finite_volume_method/kappa.npz', # <filename
                dump = '/Users/adelle/Documents/ignition_calculations/kepler/run3/h#298',
                run_name = names[i],
                )
            D.plot()
