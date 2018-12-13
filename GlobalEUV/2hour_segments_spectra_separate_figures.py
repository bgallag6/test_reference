# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 08:33:03 2018

@author: Brendan
"""

import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from scipy import fftpack
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
                 

def spec_fit( spectra ):
        
    # initialize arrays to hold parameter values
    params = np.zeros((11))
                                               
    f = freqs
    s = spectra
    
    # assign equal weights to all parts of the curve & use as fitting uncertainties
    df = np.log10(freqs[1:len(freqs)]) - np.log10(freqs[0:len(freqs)-1])
    df2 = np.zeros_like(freqs)
    df2[0:len(df)] = df
    df2[len(df2)-1] = df2[len(df2)-2]
    ds = df2
                     
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
        
    return params, fit




directory = 'S:'
date = '20130626'
wavelength = 171
n_segments = 6

cube_shape = np.load('%s/DATA/%s/%i/derotated_mmap_shape.npy' % (directory, date, wavelength))
cube = np.memmap('%s/DATA/%s/%i/derotated_mmap.npy' % (directory, date, wavelength), dtype='int16', mode='r', shape=(cube_shape[0], cube_shape[1], cube_shape[2]))

time = np.load('%s/DATA/%s/%i/time.npy' % (directory, date, wavelength))
exposure = np.load('%s/DATA/%s/%i/exposure.npy' % (directory, date, wavelength))

hmap = np.load('%s/DATA/%s/%i/param.npy' % (directory, date, wavelength))[1]
vis = np.load('%s/DATA/%s/%i/visual.npy' % (directory, date, wavelength))

font_size = 15

# determine frequency values that FFT will evaluate
if wavelength in [1600,1700]:
    time_step = 24  # add as argument in function call, or leave in as constant?
else:
    time_step = 12

t_interp = np.linspace(0, time[len(time)-1], int(time[len(time)-1]/time_step)+1)  # interpolate onto default-cadence time-grid
    
#n_segments = num_seg
n = len(t_interp)
rem = n % n_segments
freq_size = (n - rem) // n_segments

sample_freq = fftpack.fftfreq(freq_size, d=time_step)
pidxs = np.where(sample_freq > 0)
freqs = sample_freq[pidxs]

pixmed = np.zeros(cube.shape[0])  # Initialize array to hold median pixel values

x0 = [853, 300, 965, 834]
y0 = [316, 520, 865, 1413]    

x = x0[1]
y = y0[1]
    
pixmed = cube[:,y,x] / exposure  # extract timeseries + normalize by exposure time   

v_interp = np.interp(t_interp,time,pixmed)  # interpolate pixel-intensity values onto specified time grid

data = v_interp

avg_array = np.zeros((len(freqs)))  # initialize array to hold fourier powers
spectra_segs = np.zeros((6,len(freqs)))

data = data[0:len(data)-rem]  # trim timeseries to be integer multiple of n_segments
split = np.split(data, n_segments)  # create split array for each segment
t_interp = t_interp[:len(t_interp)-rem]
t_split = np.split(t_interp, n_segments)  # create split array for each segment

#"""   
#for i in range(n_segments):    
#for i in [0,1,3,4,5]: 
for i in [0,3,4,5]:         
    
  ## perform Fast Fourier Transform on each segment       
  sig = split[i]
  t = t_split[i]
  sig_fft = fftpack.fft(sig)
  #sig_fft = fftpack.rfft(sig)  # real-FFT                
  #sig_fft = accelerate.mkl.fftpack.fft(sig)  # MKL-accelerated is (2x) faster
  #sig_fft = accelerate.mkl.fftpack.rfft(sig)  # this is slightly faster
  powers = np.abs(sig_fft)[pidxs]
  norm = len(sig)
  powers = ((powers/norm)**2)*(1./(sig.std()**2))*2   # normalize the power
  spectra_segs[i] = powers
  avg_array += powers
  
  # plot timeseries & spectrum for each segment
  param, model_fit = spec_fit(powers)
  print('%0.3e, %0.3e, %0.3e, %0.3e, %0.2f, %0.2f' % (param[0], param[1], param[2], param[3], param[4], param[5]))
  
  plt.figure(figsize=(14,6))
  #ax1 = plt.subplot2grid((20,11),(1, 0), colspan=11, rowspan=9)
  ax1 = plt.subplot2grid((10,22),(0, 0), colspan=10, rowspan=10)
  ax1.set_title('Timeseries', y=1.01, fontsize=font_size)
  plt.plot(t/60, sig)
  plt.ylim(0,1700)
  ax1.set_ylabel('Intensity', fontsize=font_size-2)
  ax1.set_xlabel('Time [min]', fontsize=font_size-2)
  
  #ax2 = plt.subplot2grid((20,11),(11, 0), colspan=11, rowspan=9)
  ax2 = plt.subplot2grid((10,22),(0, 12), colspan=10, rowspan=10)
  ax2.set_title('Segment Spectrum #%i' % (i+1), y=1.01, fontsize=font_size)
  plt.loglog(freqs, powers)
  plt.loglog(freqs, model_fit)
  plt.xlim(10**-5, 10**-1)
  plt.ylim(10**-5.5, 10**0.5)
  ax2.set_ylabel('Power', fontsize=font_size-2)
  ax2.set_xlabel('Frequency [Hz]', fontsize=font_size-2)
  plt.text(0.005, 10**-0.5, r'$n$ = {0:0.2f}'.format(param[1]), fontsize=font_size)
  #plt.savefig('C:/Users/Brendan/Desktop/segment%iof%i.pdf' % ((i+1),n_segments), format='pdf', bbox_inches='tight')

#avg_array /= n_segments  # take the average of the segments
#avg_array /= 5  # take the average of the segments
avg_array /= 4  # take the average of the segments


# plot timeseries & spectrum for averaged segments
param, model_fit = spec_fit(avg_array)
print('%0.3e, %0.3e, %0.3e, %0.3e, %0.2f, %0.2f' % (param[0], param[1], param[2], param[3], param[4], param[5]))

plt.figure(figsize=(7,6))
ax2 = plt.gca()  
#ax2.set_title('Averaged Spectrum', y=1.01, fontsize=font_size)
#ax2.set_title('Averaged Spectrum w/o Segment 3', y=1.01, fontsize=font_size)
ax2.set_title('Averaged Spectrum w/o Segments 2 & 3', y=1.01, fontsize=font_size)
plt.loglog(freqs, avg_array)
plt.loglog(freqs, model_fit)
plt.xlim(10**-5, 10**-1)
plt.ylim(10**-5.5, 10**0.5)
ax2.set_ylabel('Power', fontsize=font_size-2)
ax2.set_xlabel('Frequency [Hz]', fontsize=font_size-2)
plt.text(0.005, 10**-0.5, r'$n$ = {0:0.2f}'.format(param[1]), fontsize=font_size)
#plt.text(0.00503, 10**-0.77, r'$\beta$ = {0:0.1f} [min]'.format((1./np.exp(param[4]))/60.), fontsize=font_size)
#plt.savefig('C:/Users/Brendan/Desktop/6x2hour_averaged_spectrum.pdf', format='pdf', bbox_inches='tight')
#plt.savefig('C:/Users/Brendan/Desktop/6x2hour_averaged_spectrum_minus_seg2and3.pdf', format='pdf', bbox_inches='tight')
#"""

"""
# plot timeseries & spectrum from full 12-hours
sample_freq = fftpack.fftfreq(len(t_interp), d=time_step)
pidxs = np.where(sample_freq > 0)
freqs = sample_freq[pidxs]

sig_fft = fftpack.fft(v_interp)
#sig_fft = fftpack.rfft(sig)  # real-FFT                
powers = np.abs(sig_fft)[pidxs]
norm = len(v_interp)
powers = ((powers/norm)**2)*(1./(v_interp.std()**2))*2   # normalize the power

powers *= 6

param, model_fit = spec_fit(powers)
print('%0.3e, %0.3e, %0.3e, %0.3e, %0.2f, %0.2f' % (param[0], param[1], param[2], param[3], param[4], param[5]))

plt.figure(figsize=(14,6))

ax1 = plt.subplot2grid((10,22),(0, 0), colspan=10, rowspan=10)
ax1.set_title('Timeseries', y=1.01, fontsize=font_size)
plt.plot(t_interp/60, v_interp)
plt.ylim(0,1700)
ax1.set_ylabel('Intensity', fontsize=font_size-2)
ax1.set_xlabel('Time [min]', fontsize=font_size-2)
  
#ax2 = plt.subplot2grid((20,11),(11, 0), colspan=11, rowspan=9)
ax2 = plt.subplot2grid((10,22),(0, 12), colspan=10, rowspan=10)
ax2.set_title('Averaged Spectrum', y=1.01, fontsize=font_size)
plt.loglog(freqs, powers)
plt.loglog(freqs, model_fit)
plt.xlim(10**-5, 10**-1)
plt.ylim(10**-5.5, 10**0.5)
ax2.set_ylabel('Power', fontsize=font_size-2)
ax2.set_xlabel('Frequency [Hz]', fontsize=font_size-2)
plt.text(0.005, 10**-0.5, r'$n$ = {0:0.2f}'.format(param[1]), fontsize=font_size)
#plt.text(0.00503, 10**-0.77, r'$\beta$ = {0:0.1f} [min]'.format((1./np.exp(param[4]))/60.), fontsize=font_size)
"""