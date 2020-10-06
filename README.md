# ignition-calculations


This code investigates the plausibility of a temperature gradient caused by a hotspot persisting down to
the ignition depth during an outburst of an accreting neutron star by solving a 2D heat
diffusion equation and exploring the heat transport mechanisms inside the accretion column.
There are two different version of this code that were created to ensure the accuracy of the code. The two independent codes use different methods to solve the equations, outlined in Goodwin et al (2020, submitted to MNRAS). 
The first code, written by Alexander Heger (with input from Adelle), can be run by running the runx.py scripts in the heat_substrate directory (but you will need to check the paths to dumps and opacity tables).

The second code, written by Adelle Goodwin and Frank Chambers (with input from Alex)  can be run by calling the python class object Diffuse from the finite_volume.py module in the finite_volume folder:


from diffuse import Diffuse 
# Initialisation:
D = Diffuse(Nx=50,
    ...:    ...:         Nz=80,
    ...:    ...:         Lx=0.01*1e5,
    ...:    ...:         Lz=600,
    ...:    ...:         Tcrust=1e7,
    ...:    ...:         Thotspot=1e8,
    ...:    ...:         Ttop=1e6,
    ...:    ...:          bb_flag='bb',
    ...:    ...:          Fbase=1.0e21,
    ...:    ...:         hsize=5,
    ...:    ...:          tol=1e-6,
    ...:    ...:          op=0.5,
    ...:    ...:          maxIter=300,
    ...:    ...:          run_name="sub_init",
    ...:    ...:          rhoflag="fit",
    ...:    ...:          coord="cylindrical",
    ...:    ...:          kappa_path=('/Users/adelle/Documents/ignition_calculations/cylindrical_coords/finite_volume_method/kappa.npz','kappa_fe.npz'),
    ...:    ...:          dump_path=f'/Users/adelle/Documents/ignition_calculations/kepler/run3/h#298',
    ...:          substrate = True
    ...:          )


The code saves the solution in a text file and also includes a function, D.plot_from_save('test'), 
to read in and plot an output textfile.
