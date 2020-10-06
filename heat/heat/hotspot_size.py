from heat3 import Diffuse

sizes = [0.1, 0.3, 0.6, 1.0, 1.3, 1.6, 2.0]
lengths = [0.4, 1.2, 2.4, 4.0, 5.2, 6.4, 8.0]
names = ['run3_HS_size_100m', 'run3_HS_size_300m', 'run3_HS_size_600m', 'run3_HS_size_1km', 'run3_HS_size_1.3km', 'run3_HS_size_1.6km', 'run3_HS_size_2km']

# fixed flux base boundary
# cylindrical coordinates
# hotsot size always 1/4 of total domain
# Thot = 1e8
# dump 2 minute before burst


for i in range(len(names)):
        D = Diffuse(
            nx=200,
            nz = 280,
            lx = 400000,
            ly = 1,
            lz = 600,
            x0 = 0,
            y0 = 0,
            z0 = 0,
            r = 1e6,
            geo = 'cylindrical',
            mbase = 3,
            mtop = 5, # BLACKBODY | TEMPERATURE
            surf = 'r',
            tbase = 1e8,
            fbase = 1e19,
            ttop = 1e6,
            thot = 5e7,
            xhot = 100000,
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
            dump = '/Users/adelle/Documents/ignition_calculations/kepler/run3/h#295',
            run_name = names[i],
            )
        D.plot()
