# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 15:56:37 2018

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

#"""
processed_dir = "C:/Users/Brendan/Desktop/specFit/images/processed/20120606/1600"

cube = np.load('%s/dataCube.npy' % processed_dir, mmap_mode='r')
timestamp = np.load('%s/timestamps.npy' % processed_dir)
exposure = np.load('%s/exposures.npy' % processed_dir)
#"""

"""


T = 596

z = np.zeros((5,5,T))

for i in range(0,T):
    t = float(i)
    #tmp = np.sin(4.*2*np.pi*t/T)+np.sin(5.*2*np.pi*t/T)+np.sin(6.*2*np.pi*t/T)
    #tmp = np.sin(4.*2*np.pi*t/T) + np.sin(100.*2*np.pi*t/T)
    tmp = np.sin(50.*2*np.pi*t/T)
    z[0,0,i] = tmp
    
#pixmed = z[0][0]

plt.figure()
plt.plot(timestamp, pixmed)

timeStep = 1.
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

# interpolate pixel-intensity values onto specified time grid
v_interp = np.interp(t_interp,timestamp,pixmed)  


avg_array = np.zeros((len(freqs)))

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
  
  plt.figure()
  plt.plot(freqs, powers)

avg_array /= num_segments  # average fourier power of the segments

#phase = np.arctan2(spectra.imag, spectra_array.real)

plt.figure()
plt.plot(freqs, avg_array)


   

avgTimeseries = np.average(split, axis=0)

sig_fft = fftpack.fft(avgTimeseries)               
powers = np.abs(sig_fft)[pidxs]
powers = ((powers/len(avgTimeseries))**2)*(1./(avgTimeseries.std()**2))*2  # normalize    

plt.figure()
plt.plot(freqs, powers)
#"""


"""
timestamp = np.array([i for i in range(150)])
#np.linspace
#pixmed = np.sin(100*2*np.pi*timestamp/200) + np.sin(75*2*np.pi*timestamp/200)
pixmed = np.sin(2.5*2*np.pi*timestamp/50)

plt.figure()
plt.plot(timestamp, pixmed)

timeStep = 1
num_segments = 2
 
# determine frequency values that FFT will evaluate   
n = len(timestamp)
rem = n % num_segments
freq_size = (n - rem) // num_segments

sample_freq = fftpack.fftfreq(freq_size, d=timeStep)
pidxs = np.where(sample_freq > 0)
freqs = sample_freq[pidxs]
  
avg_array = np.zeros((len(freqs)))

split = np.split(pixmed, num_segments)


# perform Fast Fourier Transform on each segment
for i in range(num_segments):     
    
  sig = split[i]
  sig_fft = fftpack.fft(sig)              
  powers = np.abs(sig_fft)[pidxs]
  powers = ((powers/len(sig))**2)*(1./(sig.std()**2))*2  # normalize
  avg_array += powers
  
  plt.figure()
  plt.plot(freqs, powers)

avg_array /= num_segments  # average fourier power of the segments

#phase = np.arctan2(spectra.imag, spectra_array.real)

plt.figure()
plt.plot(freqs, avg_array)




t_split = np.split(timestamp, num_segments)
   

plt.figure()
plt.plot(t_split[0], split[0])

plt.figure()
plt.plot(t_split[1], split[1])

avgTimeseries = np.average(split, axis=0)

plt.figure()
plt.plot(t_split[0], avgTimeseries)

sig_fft = fftpack.fft(avgTimeseries)               
powers = np.abs(sig_fft)[pidxs]
powers = ((powers/len(avgTimeseries))**2)*(1./(avgTimeseries.std()**2))*2  # normalize    

plt.figure()
plt.plot(freqs, powers)
"""


























#"""
## actual spectra
timeStep = 24.
num_segments = 1
        
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

 
x = 102
y = 76
count = 0

for ii in [-1,0]:
    for jj in [-1,0]:        
        
        # extract timeseries + normalize by exposure time
        pixmed = subcube[:,y+ii,x+jj] / exposure     
        
        # interpolate pixel-intensity values onto specified time grid
        v_interp = np.interp(t_interp,timestamp,pixmed)  
        
        avg_array = np.zeros((len(freqs)))
        
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
          phase = np.arctan2(sig_fft.imag, sig_fft.real)
          
          plt.figure()
          plt.loglog(freqs, powers)
          
          plt.figure()
          plt.plot(freqs, phase[pidxs])
        
        avg_array /= num_segments  # average fourier power of the segments
                   
        spectra_seg[count] = np.transpose(avg_array) 
        
        count += 1

## 3x3 Averaging
spectra = np.average(spectra_seg, axis=0)

#spectra = ((spectra/len(sig))**2)*(1./(sig.std()**2))*2  # normalize

#phase = np.arctan2(spectra.imag, spectra_array.real)

plt.figure()
plt.loglog(freqs, spectra)




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
#"""
