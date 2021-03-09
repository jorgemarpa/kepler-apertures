import os, sys
path = os.path.dirname(os.getcwd()) 

sys.path.append(path)
from kepler_apertures import KeplerPSF


psf = KeplerPSF(quarter=5, channel=71
            , plot=True)



