from heat3 import Diffuse
#run5:
#1 minute: 946
#40 sec: 947
#30 sec: 948
#20 sec: 949
#10 sec: 952
#0 sec: 966
dump_list = ['h#947', 'h#948', 'h#949', 'h#952', 'h#966']
T=0
names = ['run5_40sec','run5_30sec', 'run5_20sec','run5_10sec', 'run5_0sec']

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
            dump = '/Users/adelle/Documents/ignition_calculations/kepler/run5/' + dump_list[i],
            run_name = names[i],
          )

        D.plot()
