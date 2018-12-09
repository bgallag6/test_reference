# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 12:02:54 2018

@author: Brendan
"""

import numba
import numpy as np
from scipy.optimize import curve_fit as Fit
from timeit import default_timer as timer
from scipy import fftpack
from mpi4py import MPI

# define two models: without and with @numba.jit
def M2(f2, A2, n2, C2, P2, fp2, fw2):
    return A2*f2**-n2 + C2 + P2*(1./ (1.+((np.log(f2)-fp2)/fw2)**2))    

@numba.jit
def M2_jit(f2, A2, n2, C2, P2, fp2, fw2):
    return A2*f2**-n2 + C2 + P2*(1./ (1.+((np.log(f2)-fp2)/fw2)**2))    


def jitTest(n):
    # generate n spectra and add some noise 
    s0 = np.array([M2(freqs, 1e-7, 1.3, 1e-4, 1e-3, -5.1, 0.12) for i in range(n)])
    noise = np.array([np.random.randn(len(freqs))*s0[i]*0.1 for i in range(n)])
    s = s0 + noise
    
    def fit(n):
        for i in range(n):
            Fit(M2, f, s[i], bounds=(M2_low, M2_high), sigma=ds, 
                     method='dogbox', max_nfev=3000)[0]
    
    def fitJit(n):
        for i in range(n):
            Fit(M2_jit, f, s[i], bounds=(M2_low, M2_high), sigma=ds, 
                     method='dogbox', max_nfev=3000)[0]
    
    start1 = timer()
    fit(n)
    end1 = timer()
    dt1 = end1 - start1
    print("Core {0} | No @jit: {1} fits = {2:0.1f} sec.".format(rank, n, dt1), flush=True)
    
    start2 = timer()
    fitJit(n)
    end2 = timer()
    dt2 = end2 - start2
    
    dt_speed = (1-(dt2/dt1))*100
    print("Core {0} | With @jit: {1} fits = {2:0.1f} sec. = {3:0.1f}% faster".format(rank, n, dt2, dt_speed), flush=True)
    
    return dt1, dt2



comm = MPI.COMM_WORLD  # set up comms
rank = comm.Get_rank()  # Each processor gets its own "rank"
	
start = timer()

# How many processors? (pulls from "-n 4" specified in terminal) 
size = MPI.COMM_WORLD.Get_size()  

sample_freq = fftpack.fftfreq(300, d=24)
pidxs = np.where(sample_freq > 0)    
freqs = sample_freq[pidxs]

# assign equal weights to all parts of curve & use as fitting uncertainties
df = np.log10(freqs[1:len(freqs)]) - np.log10(freqs[0:len(freqs)-1])
df2 = np.zeros_like(freqs)
df2[0:len(df)] = df
df2[len(df2)-1] = df2[len(df2)-2]
ds = df2

M2_low = [0., 0.3, -0.01, 0.00001, -6.5, 0.05]
M2_high = [0.002, 6., 0.01, 0.2, -4.6, 0.8]
f = freqs

n = 500

core_dt = jitTest(n)