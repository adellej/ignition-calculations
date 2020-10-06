from heat3 import Diffuse
#run3:
#1 minute: 298
#40 sec: 301
#30 sec: 303
#20 sec: 306
#10 sec: 309
#0 sec: 321


dump_list = ['h#301', 'h#303', 'h#306', 'h#309', 'h#321']
names = ['run3_40sec','run3_30sec', 'run3_20sec','run3_10sec', 'run3_0sec']
T = 0

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
            mbase = 2,
            mtop = 5, # BLACKBODY | TEMPERATURE
            surf = 'r',
            tbase = 1e8,
            fbase = 1e19,
            ttop = 1e6,
            thot = T,
            xhot = 0,
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
            dump = '/Users/adelle/Documents/ignition_calculations/kepler/run3/' + dump_list[i],
            run_name = names[i],
            )
        D.plot()