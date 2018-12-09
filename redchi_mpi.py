# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 11:58:56 2018

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

# define Power-Law-fitting function (Model M1)
def PowerLaw(f, A, n, C):
    return A*f**-n + C
    
# define Lorentzian-fitting function
def Lorentz(f, P, fp, fw):
    return P*(1./ (1.+((np.log(f)-fp)/fw)**2))

# define combined-fitting function (Model M2)
def LorentzPowerBase(f2, A2, n2, C2, P2, fp2, fw2):
    return A2*f2**-n2 + C2 + P2*(1./ (1.+((np.log(f2)-fp2)/fw2)**2))
                 

#def spec_fit( subcube ):
def spec_fit( subcube, subcube_StdDev, subparam ):
        
  # initialize arrays to hold parameter values
  chisqrArr = np.zeros((subcube.shape[0], subcube.shape[1]))
  
  start = timer()
  T1 = 0
   
  for l in range(subcube.shape[0]):
    
    for m in range(subcube.shape[1]):
                                               
        f = freqs
        s = subcube[l][m]
               
        ds = subcube_StdDev[l][m] / np.sqrt(3)  # use 3x3 pixel-box std.dev. as fitting uncertainties   
        
        #A22, n22, C22, P22, fp22, fw22 = subparam[:6,l,m]  # unpack fitting parameters     
        """
        # create model functions from fitted parameters
        if np.isnan(P22) == True:
            #m_fit = PowerLaw(f, A22, n22, C22)  
            resids = (s - m_fit)
            chisqr =  ((resids/ds)**2).sum()
            redchisqr = chisqr / float(f.size-3)  
        else:
            m_fit = LorentzPowerBase(f, A22,n22,C22,P22,fp22,fw22)  
            resids = (s - m_fit)
            chisqr = ((resids/ds)**2).sum()
            redchisqr = chisqr / float(f.size-6)  
        
        chisqrArr[l][m] = redchisqr
        """
        
    # estimate time remaining and print to screen  (looks to be much better - not sure why had above?)
    T = timer()
    T2 = T - T1
    if l == 0:
        T_init = T - start
        T_est = T_init*(subcube.shape[0])  
        T_min, T_sec = divmod(T_est, 60)
        T_hr, T_min = divmod(T_min, 60)
        print("Currently on row %i of %i, estimated time remaining: %i:%.2i:%.2i" % (l, subcube.shape[0], T_hr, T_min, T_sec), flush=True)
    else:
        T_est2 = T2*((subcube.shape[0])-l)
        T_min2, T_sec2 = divmod(T_est2, 60)
        T_hr2, T_min2 = divmod(T_min2, 60)
        print("Currently on row %i of %i, estimated time remaining: %i:%.2i:%.2i" % (l, subcube.shape[0], T_hr2, T_min2, T_sec2), flush=True)
    T1 = T

  # print estimated and total program time to screen        
  print("Beginning Estimated time = %i:%.2i:%.2i" % (T_hr, T_min, T_sec), flush=True)
  T_act = timer() - start
  T_min3, T_sec3 = divmod(T_act, 60)
  T_hr3, T_min3 = divmod(T_min3, 60)
  print("Actual total time = %i:%.2i:%.2i" % (T_hr3, T_min3, T_sec3), flush=True) 
			
  return chisqrArr
	

comm = MPI.COMM_WORLD  # set up comms
rank = comm.Get_rank()  # Each processor gets its own "rank"
	
start = timer()

size = MPI.COMM_WORLD.Get_size()  # How many processors do we have? (pulls from "-n 4" specified in terminal execution command

directory = 'F:'
date = '20130626'
wavelength = 171

# load memory-mapped array as read-only
cube_shape = np.load('%s/DATA/Temp/%s/%i/spectra_mmap_shape.npy' % (directory, date, wavelength))
cube = np.memmap('%s/DATA/Temp/%s/%i/spectra_mmap.npy' % (directory, date, wavelength), dtype='float64', mode='r', shape=tuple(cube_shape))
cube_StdDev = np.memmap('%s/DATA/Temp/%s/%i/uncertainties_mmap.npy' % (directory, date, wavelength), dtype='float64', mode='r', shape=tuple(cube_shape))
#cube = np.load('F:/Users/Brendan/Desktop/SolarProject/data/20120923/171/20120923_171_-100_100i_-528_-132j_spectra.npy')

param = np.load('%s/DATA/Output/%s/%i/param.npy' % (directory, date, wavelength))

chunks = np.array_split(cube, size)  # Split the data based on no. of processors
chunks_StdDev = np.array_split(cube_StdDev, size)  # Split the data based on no. of processors
chunks_param = np.array_split(param, size, axis=1)

# specify which chunks should be handled by each processor
for i in range(size):
    if rank == i:
        subcube = chunks[i]
        subcube_StdDev = chunks_StdDev[i]
        subparam = chunks_param[i]

# determine frequency values that FFT will evaluate
num_freq = subcube.shape[2]  # determine nubmer of frequencies that are used
freq_size = ((num_freq)*2) + 1  # determined from FFT-averaging script
if wavelength == 1600 or wavelength == 1700:
    time_step = 24
else:
    time_step = 12
sample_freq = fftpack.fftfreq(freq_size, d=time_step)
pidxs = np.where(sample_freq > 0)
freqs = sample_freq[pidxs]

redChi2 = spec_fit( subcube, subcube_StdDev, subparam )  # Do something with the array
newData_p = comm.gather(redChi2, root=0)  # Gather all the results

# Have one node stack the results
if rank == 0:
  stack_p = np.hstack(newData_p)
 
  T_final = timer() - start
  T_min_final, T_sec_final = divmod(T_final, 60)
  T_hr_final, T_min_final = divmod(T_min_final, 60)
  print("Total program time = %i:%.2i:%.2i" % (T_hr_final, T_min_final, T_sec_final), flush=True)   
  print("Just finished region: %s %iA" % (date, wavelength), flush=True)

  np.save('%s/DATA/Output/%s/%i/redChi2' % (directory, date, wavelength), stack_p)