# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 09:29:19 2018

@author: Brendan
"""

#"""
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
                 

def spec_fit( freqs, spectra ):
        
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





#directory = 'F:/Users/Brendan/Desktop/SolarProject'
directory = 'S:'
#date = '20140822'
date = '20130625'
wavelength = 1700

#derotated = np.load('%s/DATA/Temp/%s/%i/derotated.npy' % (directory, date, wavelength))

cube_shape = np.load('%s/DATA/%s/%i/derotated_mmap_shape.npy' % (directory, date, wavelength))
derotated = np.memmap('%s/DATA/%s/%i/derotated_mmap.npy' % (directory, date, wavelength), dtype='int16', mode='r', shape=(cube_shape[0], cube_shape[1], cube_shape[2]))

#x0 = [853, 300, 965, 834]  # 171 -- 20130626
#y0 = [316, 520, 865, 1413]

x0 = [100,135]
y0 = [100,104]  

x = x0[1]
y = y0[1]

#s = derotated[:,50,100]
s = derotated[:,y,x]

if wavelength == 1600 or wavelength == 1700:
    time_step = 24  # 24 second cadence for these wavelengths
    #time_step = 12  # Jacks dataset
else:
    time_step = 12  # 12 second cadence for the others
    #time_step = 24  # for half-cadence test
    
t = np.array([time_step*i for i in range(derotated.shape[0])])  
    
sig = s
sample_freq = fftpack.fftfreq(sig.size, d=time_step)
pidxs = np.where(sample_freq > 0)
freqs = sample_freq[pidxs]
sig_fft = fftpack.fft(sig)
powers = np.abs(sig_fft)[pidxs]
norm = len(sig)  # to normalize the power
powers = ((powers/norm)**2)*(1./(sig.std()**2))*2


fig = plt.figure(figsize=(16, 12))

ax1 = plt.subplot2grid((3,7),(0, 0), colspan=7, rowspan=1)
ax1.plot(t, s)
ax1.set_title('Timeseries -- (Select segment to compute FFT)', y=1.01, fontsize=20)
ax1.set_xlim(t.min(), t.max())
ax1.set_ylim(s.min(), s.max())

ax2 = plt.subplot2grid((3,7),(1, 1), colspan=5, rowspan=2)
ax2.set_xlim(10**-5, 10**-1.5)
ax2.set_ylim(10**-7, 10**0)      
ax2.vlines((1./180.),10**-7,10**0,linestyle='dotted')  
ax2.vlines((1./300.),10**-7,10**0,linestyle='dashed') 
line2, = ax2.loglog(freqs, powers, '-')
line3, = ax2.loglog(0)


        
def onselect(tmin, tmax):
    indmin, indmax = np.searchsorted(t, (tmin, tmax))
    indmax = min(len(t) - 1, indmax)

    this_sig = s[indmin:indmax]
    this_sample_freq = fftpack.fftfreq(this_sig.size, d=time_step)
    this_pidxs = np.where(this_sample_freq > 0)
    this_freqss = this_sample_freq[this_pidxs]
    this_sig_fft = fftpack.fft(this_sig)
    this_powers = np.abs(this_sig_fft)[this_pidxs]
    this_norm = len(this_sig)  # to normalize the power
    this_powers = ((this_powers/this_norm)**2)*(1./(this_sig.std()**2))*2
    
    line2.set_data(this_freqss, this_powers)
    
    f = this_freqss   
    
    param, model_fit = spec_fit(f, this_powers)
    print('%0.3e, %0.3e, %0.3e, %0.3e, %0.2f, %0.2f' % (param[0], param[1], param[2], param[3], param[4], param[5]))
    
    line3.set_data(f, model_fit)
    
    fig.canvas.draw()

# set useblit True on gtkagg for enhanced performance
span = SpanSelector(ax1, onselect, 'horizontal', useblit=True,
                    rectprops=dict(alpha=0.5, facecolor='red'))

plt.show()
#"""
