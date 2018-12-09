# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 15:40:21 2018

@author: Brendan
"""

import numpy as np
from timeit import default_timer as timer
from scipy import fftpack
import time
import datetime
import sys
import os
import matplotlib.pyplot as plt

processed_dir = "C:/Users/Brendan/Desktop/specFit/images/processed/20120606/1600"

cube = np.load('%s/dataCube.npy' % processed_dir, mmap_mode='r')
timestamp = np.load('%s/timestamps.npy' % processed_dir)
exposure = np.load('%s/exposures.npy' % processed_dir)

timeStep = tStep = 24.
num_segments = 2
        
# interpolate timestamps onto default-cadence time-grid
t_interp = np.linspace(0, timestamp[-1], int(timestamp[-1]//timeStep)+1)  
 
# determine frequency values that FFT will evaluate   
n = len(t_interp)
rem = n % num_segments
freq_size = (n - rem) // num_segments

sample_freq = fftpack.fftfreq(freq_size, d=timeStep)
pidxs = np.where(sample_freq > 0)
freqs = sample_freq[pidxs]

## Each processor runs function on subcube, results are gathered when finished
subcube = cube

pixmed = np.empty(subcube.shape[0])
spectra_seg = np.zeros((9,len(freqs)))
full_seg = np.zeros((9,len(freqs)), dtype=complex)
std_seg = np.zeros((2,9))
 
x = 102
y = 76
count = 0

for ii in [-1,0,1]:
    for jj in [-1,0,1]:        
        
        # extract timeseries + normalize by exposure time
        pixmed = subcube[:,y+ii,x+jj] / exposure     
        
        # interpolate pixel-intensity values onto specified time grid
        v_interp = np.interp(t_interp,timestamp,pixmed)  
        
        avg_array = np.zeros((len(freqs)))
        full_array = np.zeros((len(freqs)), dtype=complex)
        
        # trim timeseries to be integer multiple of num_segments
        v_interp = v_interp[0:len(v_interp)-rem]  
        split = np.split(v_interp, num_segments)
        
        # perform Fast Fourier Transform on each segment
        for i in range(num_segments):     
            
          sig = split[i]
          sig_fft = fftpack.fft(sig)
          #sig_fft = fftpack.rfft(sig)  # real-FFT                
          powers = np.abs(sig_fft)[pidxs]
          powers = ((powers/len(sig))**2)*(1./(sig.std()**2))*2  # normalize
          avg_array += powers
          full_array += np.arctan2(np.imag(sig_fft[pidxs]), np.real(sig_fft[pidxs]))*180/np.pi
          std_seg[i][count] = sig.std()
          #full_seg += np.arctan2(np.imag(sig_fft[pidxs]), np.real(sig_fft[pidxs]))*180/np.pi
        
        avg_array /= num_segments  # average fourier power of the segments
        full_array /= num_segments
                   
        spectra_seg[count] = np.transpose(avg_array) 
        full_seg[count] = np.transpose(full_array)
        
        count += 1

"""
powers2 = 0
powers3 = 0
for i in range(9):
    for j in range(num_segments):
        powers2 += ((np.abs(full_seg[j][i])/len(sig))**2)*(1./(std_seg[j][i]**2))*2  # normalize
    powers3 += (powers2 / num_segments)
    powers2 = 0
powers3 /= 9


phase2 = 0
phase3 = 0
for i in range(9):
    for j in range(num_segments):
        phase2 += np.arctan2(np.imag(full_seg[j][i]), np.real(full_seg[j][i]))*180/np.pi
    phase3 += (phase2 / num_segments)
    phase2 = 0
phase3 /= 9


phase6 = 0
for i in range(9):
    for j in range(num_segments):
        phase6 += np.arctan2(np.imag(full_seg[j][i]), np.real(full_seg[j][i]))*180/np.pi
phase6 /= 9*num_segments

phase11 = np.average(full_seg)
"""


"""
## also try averaging full fft result, then finding phase
"""

"""
phase0 = 0
phase00 = 0
for i in range(9):
    for j in range(num_segments):
        phase0 += full_seg[j][i]
    phase00 += (phase0 / num_segments)
    phase0 = 0
phase00 /= 9

phase000 = np.arctan2(np.imag(phase00), np.real(phase00))*180/np.pi

spectra00 = (np.abs(phase00)/len(sig))**2
"""

## 3x3 Averaging
spectra = np.average(spectra_seg, axis=0)
phase = np.average(full_seg, axis=0)
#full = np.average(full_seg, axis=0)

#phase = np.arctan2(np.imag(full), np.real(full))*180/np.pi
#powers2 = (np.abs(full)/len(sig))**2

#phase = np.arctan2(spectra.imag, spectra_array.real)

plt.figure()
plt.loglog(freqs, spectra)
plt.xlim(10**-4.5, 10**-1)
plt.ylim(10**-4.5, 10**0)

plt.figure()
plt.loglog(freqs, phase)
plt.xlim(10**-4.5, 10**-1)
plt.ylim(10**-1, 10**3)

"""
plt.figure()
plt.loglog(freqs, powers3)
plt.xlim(10**-4.5, 10**-1)
plt.ylim(10**-4.5, 10**0)

plt.figure()
plt.loglog(freqs, spectra00)
plt.xlim(10**-4.5, 10**-1)
plt.ylim(10**-4.5, 10**0)

plt.figure()
plt.loglog(freqs, phase3)
plt.xlim(10**-4.5, 10**-1)
plt.ylim(10**-1, 10**3)

plt.figure()
plt.loglog(freqs, phase6)
plt.xlim(10**-4.5, 10**-1)
plt.ylim(10**-1, 10**3)

plt.figure()
plt.loglog(freqs, phase000)
plt.xlim(10**-4.5, 10**-1)
plt.ylim(10**-1, 10**3)

phase10 = np.arctan2(np.imag(full_seg[j][i]), np.real(full_seg[j][i]))*180/np.pi

plt.figure()
plt.loglog(freqs, phase10)
plt.xlim(10**-4.5, 10**-1)
plt.ylim(10**-1, 10**3)
"""





"""
count = 0

avgTimeseries = np.zeros((len(np.split(t_interp, num_segments)[0])))

for ii in [-1,0,1]:
    for jj in [-1,0,1]:        
        
        # extract timeseries + normalize by exposure time
        pixmed = subcube[:,y+ii,x+jj] / exposure     
        
        # interpolate pixel-intensity values onto specified time grid
        v_interp = np.interp(t_interp,timestamp,pixmed)  
        
        # trim timeseries to be integer multiple of num_segments
        v_interp = v_interp[0:len(v_interp)-rem]  
        split = np.split(v_interp, num_segments)
        
        avgTimeseriesTemp = np.average(split, axis=0)
        
        avgTimeseries += avgTimeseriesTemp
        
avgTimeseries /= (num_segments*9)  # average fourier power of the segments
sig_fft = fftpack.fft(avgTimeseries)               
powers = np.abs(sig_fft)[pidxs]
powers = ((powers/len(avgTimeseries))**2)*(1./(avgTimeseries.std()**2))*2  # normalize                   

plt.figure()
plt.loglog(freqs, powers)
"""