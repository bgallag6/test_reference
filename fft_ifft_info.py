# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 14:27:35 2018

@author: Brendan
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack  # doesnt work in module when called here???

"""
A = 0.5  # amplitude of cosine wave
fc = 10  # frequency of wave
phase = 30  # desired phase shift in degrees
fs = 32*fc  # sampling frequency
t = np.linspace(0, 2 - (1/fs), 2*fs)  # 2-second duration

phi = phase*np.pi / 180  # convert phase shift in degrees in radians
y = A * np.cos(2*np.pi*fc*t + phi)  # time domain signal with phase shift
"""

A = [0.5, 0.25, 0.33, 0.1]
fc = [25, 10, 30, 47]
phase = [0, 15, 60, 35]
fs = 500
t = np.linspace(0, 2 - (1/fs), 2*fs)  # 2-second duration

y = np.zeros((len(t)))

for i in range(len(A)):
    a = A[i]
    f = fc[i]
    p = phase[i]
    phi = p*np.pi / 180  # convert phase shift in degrees in radians
    #y = A * np.cos(2*np.pi*fc*t + phi)  # time domain signal with phase shift
    #y = A * np.cos(2*np.pi*fc*t + phi) + (A*2) * np.cos(2*np.pi*(fc*2)*t + 0.87) + (A/2.) * np.cos(2*np.pi*(fc*3)*t + 0.5) + (A/5.) * np.cos(2*np.pi*(fc*5)*t + 0.1) # time domain signal with phase shift
    #noise = np.random.randn(len(y))
    #y += noise  
    y += a*np.cos(2*np.pi*f*t + phi)

noise = np.random.randn(len(y))/5.
y += noise
 
#"""

"""
directory = "C:/Users/Brendan/Desktop/specFit/test/Processed/20120606/1600"
directory2 = "C:/Users/Brendan/Desktop/specFit/test/validation/Processed/20120606/1600"

cube = np.load('%s/dataCube.npy' % directory)
timestamp = np.load('%s/timestamps.npy' % directory)
exposure = np.load('%s/exposures.npy' % directory)

timeSeries = cube[:,58,102] / exposure

t = np.linspace(0, timestamp[-1], int(timestamp[-1]//24)+1) 
y = np.interp(t,timestamp,timeSeries)
fs = 1/24
"""

plt.figure()
plt.plot(t, y)
plt.title('Timeseries')
plt.xlabel('Time [sec]')
plt.ylabel('Amplitude')


# determine frequency values that FFT will evaluate   
sample_freq = fftpack.fftfreq(len(y), d=1/fs)
pidxs = np.where(sample_freq > 0)
freqs = sample_freq
#freqs = sample_freq[pidxs]

powers = fftpack.fft(y) / len(y)  # compute FFT and normalize    
#powers = np.abs(fftpack.fft(y))[pidxs]  # compute FFT
#powers = ((powers/len(y))**2)*(1./(y.std()**2))*2  # normalize    

plt.figure()
plt.plot(freqs, powers)
#plt.loglog(freqs, powers)
plt.title('Power Spectrum')
plt.xlabel('Freqeuncy [Hz]')
plt.ylabel('Power')


phase = np.arctan2(np.imag(powers), np.real(powers)) * (180/np.pi)

plt.figure()
plt.plot(freqs, phase)
plt.title('Phase Spectrum')
plt.xlabel('Freqeuncy [Hz]')
plt.ylabel('Phase [deg]')

plt.figure()
plt.boxplot(abs(powers))
plt.title('Boxplot')
plt.ylabel('Power')

powers2 = np.array(powers)
#threshold = max(abs(powers))/10000
threshold = np.percentile(abs(powers), 99)
#threshold = np.percentile(abs(powers2), 50)
powers2[abs(powers) < threshold] = 0
phase2 = np.arctan2(np.imag(powers2), np.real(powers2)) * (180/np.pi)

plt.figure()
plt.plot(freqs, phase2)
plt.title('Phase Spectrum (w/ Threshold)')
plt.xlabel('Freqeuncy [Hz]')
plt.ylabel('Phase [deg]')
