# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 13:36:01 2018

@author: Brendan
"""

import numpy as np
import scipy.signal
from pylab import *
from sunpy.map import Map
from scipy.interpolate import interp1d
from scipy import signal
import scipy.misc
import astropy.units as u
#from scipy import fftpack  # not working with this called here???
from timeit import default_timer as timer
#import accelerate  # put inside function
import glob
import matplotlib.pyplot as plt
from scipy import fftpack
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
from scipy import fftpack

import numpy as np
import scipy.signal
from scipy.interpolate import interp1d
from scipy import signal
import scipy.misc
import astropy.units as u
from scipy import fftpack
from astropy.convolution import convolve, Box1DKernel
from numpy.random import randn
from mpi4py import MPI
from scipy.stats.stats import pearsonr 


# define Power-Law-fitting function (Model M1)
def PowerLaw(f, A, n, C):
    return A*f**-n + C
    
# define Lorentzian-fitting function
def Lorentz(f, P, fp, fw):
    return P*(1./ (1.+((np.log(f)-fp)/fw)**2))

# define combined-fitting function (Model M2)
def LorentzPowerBase(f2, A2, n2, C2, P2, fp2, fw2):
    return A2*f2**-n2 + C2 + P2*(1./ (1.+((np.log(f2)-fp2)/fw2)**2))

directory = 'S:'
date = '20130626'
wavelength = 171
n_segments = 6

cube_shape = np.load('%s/DATA/%s/%i/derotated_mmap_shape.npy' % (directory, date, wavelength))
DATA = np.memmap('%s/DATA/%s/%i/derotated_mmap.npy' % (directory, date, wavelength), dtype='int16', mode='r', shape=(cube_shape[0], cube_shape[1], cube_shape[2]))

TIME = np.load('%s/DATA/%s/%i/time.npy' % (directory, date, wavelength))
Ex = np.load('%s/DATA/%s/%i/exposure.npy' % (directory, date, wavelength))

font_size = 15

# determine frequency values that FFT will evaluate
if wavelength in [1600,1700]:
    time_step = 24  # add as argument in function call, or leave in as constant?
else:
    time_step = 12

t_interp = np.linspace(0, TIME[len(TIME)-1], int((TIME[len(TIME)-1]/time_step)+1))  # interpolate onto default-cadence time-grid

#x0 = [853, 300, 965, 834]  # used for all previous
#y0 = [316, 520, 865, 1413]    

x0 = [864, 977, 847]
y0 = [318, 860, 1428]

#num_seg = np.array([1,3,6,12])  
#num_seg = np.array([2,3,6,12]) 
num_seg = np.array([1,2,3,6,12])
            
plt.figure(figsize=(13,9))
font_size = 15

for n in range(len(num_seg)):
 
    n_segments = num_seg[n]  # break data into 12 segments of equal length
    r = len(t_interp)
    rem = r % n_segments
    freq_size = (r - rem) // n_segments 
    
    sample_freq = fftpack.fftfreq(freq_size, d=time_step)
    pidxs = np.where(sample_freq > 0)
    freqs = sample_freq[pidxs]
    
    spectra_seg = np.zeros((9,len(freqs)))
    
    count = 0
    for m0 in [-1,0,1]:
        for n0 in [-1,0,1]:
            
            x = x0[0] + n0
            y = y0[0] + m0
                
            pixmed = DATA[:,y,x] / Ex  # extract timeseries + normalize by exposure time   
            
            v_interp = np.interp(t_interp,TIME,pixmed)  # interpolate pixel-intensity values onto specified time grid
            
            data = v_interp
            
            avg_array = np.zeros((len(freqs)))  # initialize array to hold fourier powers
            
            data = data[0:len(data)-rem]  # trim timeseries to be integer multiple of n_segments
            split = np.split(data, n_segments)  # create split array for each segment
            #t_interp = t_interp[:len(t_interp)-rem]
            #t_split = np.split(t_interp, n_segments)  # create split array for each segment
            
            #"""   
            for i in range(n_segments):     
                
              ## perform Fast Fourier Transform on each segment       
              sig = split[i]
              #t = t_split[i]
              sig_fft = fftpack.fft(sig)
              #sig_fft = fftpack.rfft(sig)  # real-FFT                
              #sig_fft = accelerate.mkl.fftpack.fft(sig)  # MKL-accelerated is (2x) faster
              #sig_fft = accelerate.mkl.fftpack.rfft(sig)  # this is slightly faster
              powers = np.abs(sig_fft)[pidxs]
              norm = len(sig)
              powers = ((powers/norm)**2)*(1./(sig.std()**2))*2   # normalize the power
              avg_array += powers
            
            avg_array /= n_segments  # take the average of the segments            
            #avg_array /= 5  # take the average of the segments
            
            spectra_seg[count] = avg_array
            count += 1
    
    """
    ### 3x3 Averaging
    """
    
    if n_segments == 1:
        spec_avg = spectra_seg[4]
    else:   
        spec_avg = np.average(spectra_seg, axis=0)
        #spec_std = np.std(spectra_seg, axis=0)
    
    if n_segments == 1:
        avg_array1 = spec_avg
        s = avg_array1
        freqs1 = freqs
    if n_segments == 2:
        #avg_array1 = spec_avg
        avg_array2 = spec_avg/2.
        s = avg_array2
        freqs2 = freqs
        #avg_array1 = spec_geo
    elif n_segments == 3:
        avg_array3 = spec_avg/3.
        s = avg_array3
        freqs3 = freqs
        #avg_array3 = avg_array
        #avg_array3 = spec_geo/3.
    elif n_segments == 6:
        avg_array6 = spec_avg/6.
        s = avg_array6
        freqs6 = freqs
        #avg_array6 = avg_array
        #avg_array6 = spec_geo/6.
    elif n_segments == 12:
        avg_array12 = spec_avg/12.
        s = avg_array12
        freqs12 = freqs
        #avg_array12 = avg_array
        #avg_array12 = spec_geo/12.
        

    params = np.zeros((11))
    
    f = freqs
    
    # assign equal weights to all parts of the curve & use as fitting uncertainties
    df = np.log10(freqs[1:len(freqs)]) - np.log10(freqs[0:len(freqs)-1])
    df2 = np.zeros_like(freqs)
    df2[0:len(df)] = df
    df2[len(df2)-1] = df2[len(df2)-2]
    ds = df2
    #ds = spec_std
                     
    try:
        # initial guesses for fitting parameters
        M1_low = [-0.002, 0.3, -0.01]
        M1_high = [0.002, 6., 0.01]
        nlfit_l, nlpcov_l = scipy.optimize.curve_fit(PowerLaw, f, s, bounds=(M1_low, M1_high), sigma=ds, method='dogbox')
              
    except RuntimeError:
        #print("Error M1 - curve_fit failed - %i, %i" % (l,m))  # turn off because would print too many to terminal
        pass
    
    except ValueError:
        #print("Error M1 - inf/NaN - %i, %i" % (l,m))  # turn off because would print too many to terminal
        pass

    A, n, C = nlfit_l  # unpack fitting parameters

    ## fit data to M2 model
    
    # first fit using 'dogbox' method          
    try:                                 
        M2_low = [0., 0.3, -0.01, 0.00001, -6.5, 0.05]
        M2_high = [0.002, 6., 0.01, 0.2, -4.6, 0.8]
        
        nlfit_gp, nlpcov_gp = scipy.optimize.curve_fit(LorentzPowerBase, f, s, bounds=(M2_low, M2_high), sigma=ds, method='dogbox', max_nfev=3000)
        #nlfit_gp, nlpcov_gp = scipy.optimize.curve_fit(LorentzPowerBase, f, s, p0 = [A,n,C,0.1,-5.55,0.425], bounds=(M2_low, M2_high), sigma=ds, method='dogbox', max_nfev=3000)
    
    except RuntimeError:
        #print("Error M2 - curve_fit failed - %i, %i" % (l,m))  # turn off because would print too many to terminal
        pass
    
    except ValueError:
        #print("Error M2 - inf/NaN - %i, %i" % (l,m))  # turn off because would print too many to terminal
        pass
    
    A2, n2, C2, P2, fp2, fw2 = nlfit_gp  # unpack fitting parameters
    
    # next fit using default 'trf' method
    try:
        
        nlfit_gp2, nlpcov_gp2 = scipy.optimize.curve_fit(LorentzPowerBase, f, s, p0 = [A2, n2, C2, P2, fp2, fw2], bounds=(M2_low, M2_high), sigma=ds, max_nfev=3000)
        #nlfit_gp2, nlpcov_gp2 = scipy.optimize.curve_fit(LorentzPowerBase, f, s, bounds=(M2_low, M2_high), sigma=ds, max_nfev=3000)
   
    except RuntimeError:
        #print("Error M2 - curve_fit failed - %i, %i" % (l,m))  # turn off because would print too many to terminal
        pass
    
    except ValueError:
        #print("Error M2 - inf/NaN - %i, %i" % (l,m))  # turn off because would print too many to terminal
        pass
    
    A22, n22, C22, P22, fp22, fw22 = nlfit_gp2  # unpack fitting parameters     
                   
    # create model functions from fitted parameters
    m1_fit = PowerLaw(f, A, n, C)        
    m2_fit2 = LorentzPowerBase(f, A22,n22,C22,P22,fp22,fw22)      
    
    residsM1 = (s - m1_fit)
    chisqrM1 =  ((residsM1/ds)**2).sum()
    redchisqrM1 = chisqrM1 / float(f.size-3)  
    
    residsM22 = (s - m2_fit2)
    chisqrM22 = ((residsM22/ds)**2).sum()
    redchisqrM22 = chisqrM22 / float(f.size-6)         
    
    f_test2 = ((chisqrM1-chisqrM22)/(6-3))/((chisqrM22)/(f.size-6))
    
    amp_scale2 = P22 / PowerLaw(np.exp(fp22), A22, n22, C22)  # to extract the lorentzian-amplitude scaling factor
    
    if chisqrM1 > chisqrM22:
        rval = pearsonr(m2_fit2, s)[0]  # calculate r-value correlation coefficient
        rollover = (1. / ((C22 / A22)**(-1. / n22))) / 60.
        
        # populate array with M2 parameters
        params[0] = A22
        params[1] = n22
        params[2] = C22
        params[3] = P22
        params[4] = fp22
        params[5] = fw22
        params[6] = f_test2
        params[7] = amp_scale2
        params[8] = rval
        params[9] = rollover
        params[10] = redchisqrM22
        
        fit = m2_fit2
        
    else:
        rval = pearsonr(m1_fit, s)[0]
        rollover = (1. / ((C / A)**(-1. / n))) / 60.
        
        # populate array with M1 parameters
        params[0] = A
        params[1] = n
        params[2] = C
        params[3] = np.NaN
        params[4] = np.NaN
        params[5] = np.NaN
        params[6] = np.NaN
        params[7] = np.NaN
        params[8] = rval
        params[9] = rollover
        params[10] = redchisqrM1
        
        fit = m1_fit
    
    print('%0.3e, %0.3f, %0.3e, %0.3e, %0.2f, %0.2f' % (params[0], params[1], params[2], params[3], (1./np.exp(params[4]))/60., params[5]))
    fwhm = (1. / (np.exp(params[4]+params[5]) - np.exp(params[4]-params[5])))/60.
    
    if n_segments == 1:
        fit1 = fit
    if n_segments == 2:
        fit2 = fit
        ax1 = plt.subplot2grid((21,22),(0, 0), colspan=10, rowspan=10)
        #ax1.set_title('(1) 12-Hour Segment', fontsize=font_size)
        ax1.set_title('(2) 6-Hour Segments', fontsize=font_size)
        ax1.set_ylabel('Power', fontsize=font_size-2)
        plt.loglog(freqs, s)
        plt.loglog(freqs, fit)
        plt.xlim(10**-5, 10**-1)
        plt.ylim(10**-5.5, 10**0.5) 
        plt.text(10**-2.53,10**-0.5,'n = %0.2f' % params[1], fontsize=font_size)
        plt.text(10**-2.53,10**-0.95,'β = %0.2f [min]' % ((1./np.exp(params[4]))/60.), fontsize=font_size) 
        plt.text(10**-3,10**-1.4,r'FWHM = %0.2f [min]' % fwhm, fontsize=font_size)
        #avg_array1 = spec_geo
    elif n_segments == 3:
        fit3 = fit
        ax2 = plt.subplot2grid((21,22),(0, 12), colspan=10, rowspan=10)
        ax2.set_title('(3) 4-Hour Segments', fontsize=font_size)
        plt.loglog(freqs, s)
        plt.loglog(freqs, fit)
        plt.xlim(10**-5, 10**-1)
        plt.ylim(10**-5.5, 10**0.5) 
        plt.text(10**-2.53,10**-0.5,'n = %0.2f' % params[1], fontsize=font_size)
        plt.text(10**-2.53,10**-0.95,'β = %0.2f [min]' % ((1./np.exp(params[4]))/60.), fontsize=font_size) 
        plt.text(10**-3,10**-1.4,r'FWHM = %0.2f [min]' % fwhm, fontsize=font_size)
        #avg_array3 = avg_array
        #avg_array3 = spec_geo/3.
    elif n_segments == 6:
        fit6 = fit
        ax3 = plt.subplot2grid((21,22),(12, 0), colspan=10, rowspan=10)
        ax3.set_title('(6) 2-Hour Segments', fontsize=font_size)
        ax3.set_ylabel('Power', fontsize=font_size-2)
        ax3.set_xlabel('Frequency [Hz]', fontsize=font_size-2)
        plt.loglog(freqs, s)
        plt.loglog(freqs, fit)
        plt.xlim(10**-5, 10**-1)
        plt.ylim(10**-5.5, 10**0.5) 
        plt.text(10**-2.53,10**-0.5,'n = %0.2f' % params[1], fontsize=font_size)
        plt.text(10**-2.53,10**-0.95,'β = %0.2f [min]' % ((1./np.exp(params[4]))/60.), fontsize=font_size) 
        plt.text(10**-3,10**-1.4,r'FWHM = %0.2f [min]' % fwhm, fontsize=font_size)
        #avg_array6 = avg_array
        #avg_array6 = spec_geo/6.
    elif n_segments == 12:
        fit12 = fit
        ax4 = plt.subplot2grid((21,22),(12, 12), colspan=10, rowspan=10)
        ax4.set_title('(12) 1-Hour Segments', fontsize=font_size)
        ax4.set_xlabel('Frequency [Hz]', fontsize=font_size-2)
        plt.loglog(freqs, s)
        plt.loglog(freqs, fit)
        plt.xlim(10**-5, 10**-1)
        plt.ylim(10**-5.5, 10**0.5) 
        plt.text(10**-2.53,10**-0.5,'n = %0.2f' % params[1], fontsize=font_size)
        plt.text(10**-2.53,10**-0.95,'β = %0.2f [min]' % ((1./np.exp(params[4]))/60.), fontsize=font_size) 
        plt.text(10**-3,10**-1.4,r'FWHM = %0.2f [min]' % fwhm, fontsize=font_size)
        #avg_array12 = avg_array
        #avg_array12 = spec_geo/12.
        #plt.savefig('C:/Users/Brendan/Desktop/spectra_time_segmenting_3x3B.pdf', format='pdf', bbox_inches='tight')

#spectra_seg[jj-245+(ii-163)*3] = powers  # construct 3D array with averaged FFTs from each pixel
#spectra_std = np.std(spectra_seg, axis=0)

#plt.figure(figsize=(15,15))
#plt.loglog(freqs,avg_array)
#plt.ylim(10**-6.5,10**0)
        
plt.rcParams["font.family"] = "Times New Roman"
font_size = 27
    
plt.figure(figsize=(13,9))
ax = plt.gca()  
plt.title('Comparison of Temporal Averaging Methods', y=1.01, fontsize=25)
plt.loglog(freqs1,avg_array1, 'k', linewidth=1.7, label='(1) 12-Hour Segment')
plt.loglog(freqs2,avg_array2, color='orange', linewidth=1.7, label='(2) 6-Hour Segments')
plt.loglog(freqs3,avg_array3, 'b', linewidth=1.7, label='(3) 4-Hour Segments')
plt.loglog(freqs6,avg_array6, 'g', linewidth=1.7, label='(6) 2-Hour Segments')
plt.loglog(freqs12,avg_array12, 'r', linewidth=1.7, label='(12) 1-Hour Segments')
plt.loglog(freqs1,fit1, 'k--',  linewidth=1.7)
plt.loglog(freqs2,fit2,  color='orange', linestyle='--', linewidth=1.7)
plt.loglog(freqs3,fit3, 'b--', linewidth=1.7)
plt.loglog(freqs6,fit6, 'g--', linewidth=1.7)
plt.loglog(freqs12,fit12, 'r--', linewidth=1.7)
#plt.ylim(10**-6.5,10**0)
#plt.xlim(10**-5.,10**-1.3)
plt.xlim(10**-5, 10**-1.2)
plt.ylim(10**-6, 10**0)
plt.xticks(fontsize=font_size, fontname="Times New Roman")
plt.yticks(fontsize=font_size, fontname="Times New Roman")
ax.set_ylabel('Power', fontsize=font_size-2)
ax.set_xlabel('Frequency [Hz]', fontsize=font_size-2)
plt.tick_params(axis='both', which='major', pad=10)
legend = plt.legend(loc='lower left', prop={'size':font_size}, labelspacing=0.35)
for label in legend.get_lines():
    label.set_linewidth(3.0)  # the legend line width
#plt.savefig('C:/Users/Brendan/Desktop/spectra_temporal_averaging_methods_3x3B.pdf', format='pdf', bbox_inches='tight')

#np.save('C:/Users/Brendan/Desktop/spec_array1_jack.npy', avg_array1)
#np.save('C:/Users/Brendan/Desktop/spec_array3_jack.npy', avg_array3)
#np.save('C:/Users/Brendan/Desktop/spec_array6_jack.npy', avg_array6)
#np.save('C:/Users/Brendan/Desktop/spec_array12_jack.npy', avg_array12)