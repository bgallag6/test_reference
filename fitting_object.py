# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 23:12:39 2018

@author: Brendan
"""

"""
######################
# run with:
# $ mpiexec -n # python part3_spec_fit_mpi.py    (# = number of processors)
######################
"""


from timeit import default_timer as timer
import numpy as np
import scipy.signal
import scipy.misc
from scipy import fftpack
from mpi4py import MPI
from scipy.stats.stats import pearsonr 
import yaml
import matplotlib.pyplot as plt

# define Power-Law-fitting function (Model M1)
def PowerLaw(f, A, n, C):
    return A*f**-n + C
    
# define Lorentzian-fitting function
def Lorentz(f, P, fp, fw):
    #return P*(1./ (1.+((np.log(f)-fp)/fw)**2))
    return P*(1./ (1.+((f-fp)/fw)**2))

# define combined-fitting function (Model M2)
def LorentzPowerBase(f2, A2, n2, C2, P2, fp2, fw2):
    #return A2*f2**-n2 + C2 + P2*(1./ (1.+((np.log(f2)-fp2)/fw2)**2))
    return A2*f2**-n2 + C2 + P2*(1./ (1.+((f2-fp2)/fw2)**2))

class Fit():      
    def __init__(self, s):
        self.spec = s
#        self.unc = ds
 
    def M1(self):
        try: nlfit_l, nlpcov_l = scipy.optimize.curve_fit(PowerLaw, f, self.spec, bounds=(M1_low, M1_high), sigma=ds, method='dogbox')                  
        except RuntimeError: pass      
        except ValueError: pass
    
        return nlfit_l
    
    def M2(self):
        # first fit using 'dogbox' method          
        try: nlfit_gp, nlpcov_gp = scipy.optimize.curve_fit(LorentzPowerBase, f, self.spec, bounds=(M2_low, M2_high), sigma=ds, method='dogbox', max_nfev=3000)   
        except RuntimeError: pass
        except ValueError: pass
        
        #A2, n2, C2, P2, fp2, fw2 = nlfit_gp  # unpack fitting parameters
        
        # next fit using default 'trf' method
        try: nlfit_gp2, nlpcov_gp2 = scipy.optimize.curve_fit(LorentzPowerBase, f, self.spec, p0 = nlfit_gp, bounds=(M2_low, M2_high), sigma=ds, max_nfev=3000)   
        except RuntimeError: pass
        except ValueError: pass
    
        return nlfit_gp2

                 

#def spec_fit( subcube ):
def spec_fit( subcube ):
   
  for l in range(1):
  #for l in range(0,15):
    
    for m in range(1):
    #for m in range(0,20):
                                               
        f = freqs
        s = subcube[l][m]
        #ds = subcube_StdDev[l][m]  # use 3x3 pixel-box std.dev. as fitting uncertainties  
        
        ### fit data to models using SciPy's Levenberg-Marquart method
        m1_params = Fit(s).M1()
        A, n, C = m1_params  # unpack fitting parameters
        
        m2_params = Fit(s).M2()
        A22, n22, C22, P22, fp22, fw22 = m2_params  # unpack fitting parameters 
           
        # create model functions from fitted parameters
        m1_fit = PowerLaw(f, A, n, C)        
        m2_fit2 = LorentzPowerBase(f, A22,n22,C22,P22,fp22,fw22) 
        
        return s, m1_fit, m2_fit2
        
        

directory = 'S:'
date = '20001111'
wavelength = 1600

# load memory-mapped array as read-only
#cube_shape = np.load('%s/DATA/Temp/%s/%i/spectra_mmap_shape.npy' % (directory, date, wavelength))
#cube = np.memmap('%s/DATA/Temp/%s/%i/spectra_mmap.npy' % (directory, date, wavelength), dtype='float64', mode='r', shape=tuple(cube_shape))
#cube_StdDev = np.memmap('%s/DATA/Temp/%s/%i/3x3_stddev_mmap.npy' % (directory, date, wavelength), dtype='float64', mode='r', shape=tuple(cube_shape))
cube = np.load('%s/DATA/%s/%i/specCube.npy' % (directory, date, wavelength))
#cube = np.load('F:/Users/Brendan/Desktop/SolarProject/data/20120923/171/20120923_171_-100_100i_-528_-132j_spectra.npy')

M1_low = [-0.002, 0.3, -0.01]
M1_high = [0.002, 6., 0.01]
M2_low = [0., 0.3, -0.01, 0., (1./660.), -0.01]
M2_high = [0.002, 6., 0.01, 0.2, (1./100.), 0.01]

# determine frequency values that FFT will evaluate
num_freq = cube.shape[2]  # determine nubmer of frequencies that are used
freq_size = ((num_freq)*2) + 1  # determined from FFT-averaging script
if wavelength == 1600 or wavelength == 1700:
    time_step = 24
else:
    time_step = 12
sample_freq = fftpack.fftfreq(freq_size, d=time_step)
pidxs = np.where(sample_freq > 0)
freqs = sample_freq[pidxs]
f = freqs

# assign equal weights to all parts of the curve & use as fitting uncertainties
df = np.log10(freqs[1:len(freqs)]) - np.log10(freqs[0:len(freqs)-1])
df2 = np.zeros_like(freqs)
df2[0:len(df)] = df
df2[len(df2)-1] = df2[len(df2)-2]
ds = df2

s, m1, m2 = spec_fit(cube)

plt.figure()
plt.loglog(f,s)
plt.loglog(f,m1)
plt.loglog(f,m2)