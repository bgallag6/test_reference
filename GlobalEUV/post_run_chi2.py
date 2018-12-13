# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 10:37:27 2018

@author: Brendan
"""

import numpy as np
from scipy import fftpack

# define combined-fitting function (Model M2)
def LorentzPowerBase(f2, A2, n2, C2, P2, fp2, fw2):
    return A2*f2**-n2 + C2 + P2*(1./ ((np.pi*fw2)*(1.+((np.log(f2)-fp2)/fw2)**2))) 

directory = 'F:'
date = '20130626'
wavelength = 1700

param = np.load('%s/DATA/Output/%s/%i/param.npy' % (directory, date, wavelength))
cube_shape = np.load('%s/DATA/Temp/%s/%i/spectra_mmap_shape.npy' % (directory, date, wavelength))
spectra = np.memmap('%s/DATA/Temp/%s/%i/spectra_mmap.npy' % (directory, date, wavelength), dtype='float64', mode='r', shape=(cube_shape[0], cube_shape[1], cube_shape[2]))
stddev = np.memmap('%s/DATA/Temp/%s/%i/uncertainties_mmap.npy' % (directory, date, wavelength), dtype='float64', mode='r', shape=(cube_shape[0], cube_shape[1], cube_shape[2]))

### determine frequency values that FFT will evaluate
if wavelength == 1600 or wavelength == 1700:
    time_step = 24
else:
    time_step = 12
freq_size = (cube_shape[2]*2)+1
sample_freq = fftpack.fftfreq(freq_size, d=time_step)
pidxs = np.where(sample_freq > 0)    
freqs = sample_freq[pidxs] 

redchisqr = np.zeros((spectra.shape[0],spectra.shape[1]))

for m in range(spectra.shape[0]):    
    for n in range(spectra.shape[1]):
        s = np.zeros((spectra.shape[2]))  
        stD = np.zeros((spectra.shape[2])) 
        s[:] = spectra[m][n][:]
        stD[:] = stddev[m][n][:]
        m2_fit = LorentzPowerBase(freqs, *param[:6,m,n])
        resids = s - m2_fit
        chisqr = ((resids/stD)**2).sum()
        redchisqr[m][n] = chisqr / float(freqs.size - 6)

#np.save('%s/DATA/Output/%s/%i/redchi2_stddev' % (directory, date, wavelength), redchisqr)