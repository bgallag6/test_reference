# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 21:15:30 2018

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

directory = "C:/Users/Brendan/Desktop/specFit/images/processed/20120606/1600"
#directory2 = "C:/Users/Brendan/Desktop/specFit/test/validation/Processed/20120606/1600"

cube = np.load('%s/dataCube.npy' % directory)
timestamp = np.load('%s/timestamps.npy' % directory)
exposure = np.load('%s/exposures.npy' % directory)
vis = np.load('%s/visual.npy' % directory)
#vis = np.load('%s/visual.npy' % directory2)

timeSeries = cube[58,102] / exposure

t_interp = np.linspace(0, timestamp[-1], int(timestamp[-1]//24)+1) 

v_interp = np.interp(t_interp,timestamp,timeSeries)

plt.figure()
plt.plot(t_interp, v_interp)

freq_size = len(t_interp)

# determine frequency values that FFT will evaluate   
sample_freq = fftpack.fftfreq(freq_size, d=24)
pidxs = np.where(sample_freq > 0)

if freq_size % 2 == 0: # Even time series length. Keep f = -0.5 value.
    pidxs = np.append(pidxs,[pidxs[0][-1]+1])

freqs = sample_freq[pidxs]

if freq_size % 2 == 0:
    freqs[-1] = -freqs[-1]

sig_fft = fftpack.fft(v_interp)             
powers = np.abs(sig_fft)[pidxs]
powers = ((powers/len(v_interp))**2)*(1./(v_interp.std()**2))*2  # normalize

fftA = np.real(sig_fft)
fftB = np.imag(sig_fft)

plt.figure()
plt.loglog(freqs, powers)

#"""
ifftTS = np.fft.ifft(sig_fft)

plt.figure()
plt.plot(t_interp, ifftTS)

angle = np.angle(sig_fft)

plt.figure()
plt.plot(t_interp, angle)

phase = np.arctan2(fftB, fftA)

m2 = GaussPowerBase(freqs, 1e-7, 1.3, 1e-4, 1e-3, -5.2, 0.15)

plt.figure()
plt.loglog(freqs, m2)

spec_real = np.zeros((len(m2)), dtype=complex)
for i in range(len(m2)):
    rand = np.random.random()
    ai = np.sqrt(rand*m2[i]**2)
    bi = np.sqrt(m2[i]**2 - ai**2)
    spec_real[i] = complex(ai, bi)
    
spec_full = np.append(spec_real, np.flipud(spec_real))

ifft_spec = np.fft.ifft(spec_full)

plt.figure()
plt.plot(t_interp, ifft_spec)

#"""
# determine frequency values that FFT will evaluate   
sample_freq1 = fftpack.fftfreq(len(ifft_spec), d=24)
pidxs1 = np.where(sample_freq1 > 0)

if freq_size % 2 == 0: # Even time series length. Keep f = -0.5 value.
    pidxs1 = np.append(pidxs1,[pidxs1[0][-1]+1])

freqs1 = sample_freq1[pidxs1]

if freq_size % 2 == 0:
    freqs1[-1] = -freqs1[-1]

sig_fft1 = fftpack.fft(ifft_spec)             
powers1 = np.abs(sig_fft1)[pidxs1]
#powers = ((powers/len(ifft_spec))**2)*(1./(ifft_spec.std()**2))*2  # normalize

plt.figure()
plt.loglog(freqs1, powers1)
#"""



#Set the phase = 0 by setting either a or b = 0 (with a = 0, will get what I was plotting when I did a power law) or 
#set phase = random to get something that looks more like real signal: 
#if the FT at frequency f is a(f) + j*b(f), let a be a random number between 0 and sqrt(M2(f)). Then b(f) = +sqrt(M2(f)-a(f)^2).
#"""


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
# phase stuff
# determine frequency values that FFT will evaluate   
sample_freq = fftpack.fftfreq(len(t_interp), d=24)
freqs = sample_freq[pidxs]

sig_fft = fftpack.fft(v_interp)[pidxs]

sig_fft2 = np.array(sig_fft)
#threshold = max(abs(sig_fft))/10000
threshold = 60
sig_fft2[abs(sig_fft) < threshold] = 0
#phase = np.arctan2(np.imag(sig_fft2), np.real(sig_fft2))*180/np.pi
phase = np.arctan2(np.imag(sig_fft), np.real(sig_fft))*180/np.pi

plt.figure()
plt.hist(sig_fft)

plt.figure()
plt.plot(freqs,phase)

plt.figure()
plt.loglog(freqs, phase)
"""