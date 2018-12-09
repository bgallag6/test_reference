# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 15:03:53 2018

@author: Brendan
"""

import numpy as np
import scipy.signal
from scipy.interpolate import interp1d
from scipy import signal
import scipy.misc
import astropy.units as u
from scipy import fftpack  # doesnt work in module when called here???
from astropy.convolution import convolve, Box1DKernel
from numpy.random import randn
from mpi4py import MPI
import matplotlib.pyplot as plt

from scipy import fftpack    

# define Power-Law-fitting function (Model M1)
def PowerLaw(f, A, n, C):
    return A*f**-n + C
    
# define Gaussian-fitting function
def Gauss(f, P, fp, fw):
    return P*np.exp(-0.5*(((np.log(f))-fp)/fw)**2)

# define combined-fitting function (Model M2)
def GaussPowerBase(f2, A2, n2, C2, P2, fp2, fw2):
    return A2*f2**-n2 + C2 + P2*np.exp(-0.5*(((np.log(f2))-fp2)/fw2)**2)

directory = "C:/Users/Brendan/Desktop/specFit/test/Processed/20120606/1600"
directory2 = "C:/Users/Brendan/Desktop/specFit/test/validation/Processed/20120606/1600"

cube = np.load('%s/dataCube.npy' % directory)
timestamp = np.load('%s/timestamps.npy' % directory)
exposure = np.load('%s/exposures.npy' % directory)
vis = np.load('%s/visual.npy' % directory2)

timeSeries = cube[:,65,81] / exposure

t_interp = np.linspace(0, timestamp[-1], int(timestamp[-1]//24)+1) 

v_interp = np.interp(t_interp,timestamp,timeSeries)

plt.figure()
plt.plot(t_interp, v_interp)

# determine frequency values that FFT will evaluate   
sample_freq = fftpack.fftfreq(len(t_interp), d=24)
pidxs = np.where(sample_freq > 0)
freqs = sample_freq[pidxs]

sig_fft = fftpack.fft(v_interp)             
powers = np.abs(sig_fft)[pidxs]
powers = ((powers/len(v_interp))**2)*(1./(v_interp.std()**2))*2  # normalize

fftA = np.real(sig_fft)
fftB = np.imag(sig_fft)

plt.figure()
plt.loglog(freqs, powers)

"""
ifftTS = np.fft.ifft(sig_fft)

plt.figure()
plt.plot(t_interp, ifftTS)

angle = np.angle(sig_fft)

plt.figure()
plt.plot(t_interp, angle)

phase = np.arctan2(fftB, fftA)
"""


"""
# determine frequency values that FFT will evaluate   
sample_freq = fftpack.fftfreq(len(t_interp), d=24)
pidxs = np.where(sample_freq > 0)
freqs = sample_freq

sig_fft = fftpack.fft(v_interp)

sig_fft2 = np.array(sig_fft)
#threshold = max(abs(sig_fft))/10000
threshold = 60
sig_fft2[abs(sig_fft) < threshold] = 0
phase = np.arctan2(np.imag(sig_fft2), np.real(sig_fft2))*180/np.pi

plt.figure()
plt.plot(freqs,phase)
"""


"""
# determine frequency values that FFT will evaluate   
sample_freq = fftpack.fftfreq(len(t_interp), d=24)
freqs = sample_freq[pidxs]

sig_fft = fftpack.fft(v_interp)[pidxs]

sig_fft2 = np.array(sig_fft)
#threshold = max(abs(sig_fft))/10000
threshold = 60
sig_fft2[abs(sig_fft) < threshold] = 0
phase = np.arctan2(np.imag(sig_fft2), np.real(sig_fft2))*180/np.pi

plt.figure()
plt.plot(freqs,phase)

plt.figure()
plt.loglog(freqs, phase)
"""