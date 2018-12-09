# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 07:31:20 2018

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
    
# define combined-fitting function (Model M2)
def LorentzPowerBase(f2, A2, n2, C2, P2, fp2, fw2):
    return A2*f2**-n2 + C2 + P2*(1./ (1.+((np.log(f2)-fp2)/fw2)**2))  

# define Lorentzian-fitting function
def Lorentz(f, P, fp, fw):
    return P*(1./ (1.+((np.log(f)-fp)/fw)**2)) 

def KappaPowerBase(f2, A2, n2, C2, P2, fp2, fw2):
    return A2*f2**-n2 + C2 + P2*((1 + (f2**2 / (fp2*fw2**2)))**(-(fp2+1)/2))

def Kappa(f, P, fp, fw):
    return P*((1 + (f**2 / (fp*fw**2)))**(-(fp+1)/2))

def GaussPowerBase(f2, A2, n2, C2, P2, fp2, fw2):
    return A2*f2**-n2 + C2 + P2*np.exp(-0.5*(((np.log(f2))-fp2)/fw2)**2)

def Gauss(f, P, fp, fw):
    return P*np.exp(-0.5*(((np.log(f))-fp)/fw)**2)
 

"""
directory = 'S:'
date = '20130626'
wavelength = 171
n_segments = 6
"""

directory = 'F:'
date = '20140818'
wavelength = 1600

#cube_shape = np.load('%s/DATA/Temp/%s/%i/derotated_mmap_shape.npy' % (directory, date, wavelength))
#DATA = np.memmap('%s/DATA/Temp/%s/%i/derotated_mmap.npy' % (directory, date, wavelength), dtype='int16', mode='r', shape=(cube_shape[0], cube_shape[1], cube_shape[2]))
DATA = np.load('%s/DATA/Temp/%s/%i/derotated.npy' % (directory, date, wavelength))

TIME = np.load('%s/DATA/Temp/%s/%i/time.npy' % (directory, date, wavelength))
Ex = np.load('%s/DATA/Temp/%s/%i/exposure.npy' % (directory, date, wavelength))

font_size = 15

# determine frequency values that FFT will evaluate
if wavelength in [1600,1700]:
    time_step = 24  # add as argument in function call, or leave in as constant?
else:
    time_step = 12

t_interp = np.linspace(0, TIME[len(TIME)-1], int((TIME[len(TIME)-1]/time_step)+1))  # interpolate onto default-cadence time-grid

#x0 = [853, 300, 965, 834]
#y0 = [316, 520, 865, 1413]    

"""
# 171 20130626
x0 = [1387, 1560, 857, 668, 783]
y0 = [412, 650, 1222, 1251, 1376]

y0 = 1660 - np.array(y0)
"""

# 1600 20140818
x0 = [116]
y0 = [27]

k = 0
 
n_segments = 6
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
        
        x = x0[k] + n0
        y = y0[k] + m0
            
        pixmed = DATA[:,y,x] / Ex  # extract timeseries + normalize by exposure time   
        
        v_interp = np.interp(t_interp,TIME,pixmed)  # interpolate pixel-intensity values onto specified time grid
        
        data = v_interp
        
        avg_array = np.zeros((len(freqs)))  # initialize array to hold fourier powers
        
        data = data[0:len(data)-rem]  # trim timeseries to be integer multiple of n_segments
        split = np.split(data, n_segments)  # create split array for each segment
        t_interp = t_interp[:len(t_interp)-rem]
        t_split = np.split(t_interp, n_segments)  # create split array for each segment
        
        #"""   
        for i in range(n_segments):     
            
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
          avg_array += powers
        
        avg_array /= n_segments  # take the average of the segments            
        #avg_array /= 5  # take the average of the segments
        
        spectra_seg[count] = avg_array
        count += 1

"""
### 3x3 Averaging
"""

spec_avg = np.average(spectra_seg, axis=0)

s = spec_avg
    

params = np.zeros((11))

f = freqs

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
    
    nlfit_gp, nlpcov_gp = scipy.optimize.curve_fit(GaussPowerBase, f, s, bounds=(M2_low, M2_high), sigma=ds, method='dogbox', max_nfev=3000)
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
    
    nlfit_gp2, nlpcov_gp2 = scipy.optimize.curve_fit(GaussPowerBase, f, s, p0 = [A2, n2, C2, P2, fp2, fw2], bounds=(M2_low, M2_high), sigma=ds, max_nfev=3000)
    #nlfit_gp2, nlpcov_gp2 = scipy.optimize.curve_fit(LorentzPowerBase, f, s, bounds=(M2_low, M2_high), sigma=ds, max_nfev=3000)
   
except RuntimeError:
    #print("Error M2 - curve_fit failed - %i, %i" % (l,m))  # turn off because would print too many to terminal
    pass

except ValueError:
    #print("Error M2 - inf/NaN - %i, %i" % (l,m))  # turn off because would print too many to terminal
    pass

A22, n22, C22, P22, fp22, fw22 = nlfit_gp2  # unpack fitting parameters     
               
# create model functions from fitted parameters
m1_fit = PowerLaw(f, A22, n22, C22)        
gauss = Gauss(f, P22,fp22,fw22) 
m2_fit2 = GaussPowerBase(f, A22,n22,C22,P22,fp22,fw22)      

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

fwhm = (1. / (np.exp(params[4]+params[5]) - np.exp(params[4]-params[5])))/60.

plt.rcParams["font.family"] = "Times New Roman"
font_size = 17
    
plt.figure(figsize=(22,7))
ax1 = plt.subplot2grid((10,32),(0, 0), colspan=10, rowspan=10)
ax1.set_title('Gaussian Function', fontsize=font_size)
plt.loglog(freqs, s, 'k')
plt.loglog(freqs, fit, color='purple')
plt.loglog(freqs, m1_fit, color='green', linestyle='dashed')
plt.loglog(freqs, gauss, color='green', linestyle='dashed')
plt.xlim(10**-4.5, 10**-1.)
plt.ylim(10**-4.5, 10**0) 
plt.text(10**-2.1,10**-0.5,'n = %0.2f' % params[1], fontsize=font_size)
plt.text(10**-2.1,10**-0.75,'β = %0.2f [min]' % ((1./np.exp(params[4]))/60.), fontsize=font_size) 
plt.text(10**-2.51,10**-1.,r'FWHM = %0.2f [min]' % fwhm, fontsize=font_size)
#plt.savefig('C:/Users/Brendan/Desktop/spectra_temporal_averaging_methods_point.pdf', format='pdf', bbox_inches='tight')








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
m1_fit = PowerLaw(f, A22, n22, C22)   
lorentz = Lorentz(f, P22,fp22,fw22) 
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
    
ax2 = plt.subplot2grid((10,32),(0, 11), colspan=10, rowspan=10)
ax2.set_title('Lorentzian Function', fontsize=font_size)
plt.loglog(freqs, s, 'k')
plt.loglog(freqs, fit, color='purple')
plt.loglog(freqs, m1_fit, color='green', linestyle='dashed')
plt.loglog(freqs, lorentz, color='green', linestyle='dashed')
plt.xlim(10**-4.5, 10**-1.)
plt.ylim(10**-4.5, 10**0) 
plt.text(10**-2.1,10**-0.5,'n = %0.2f' % params[1], fontsize=font_size)
plt.text(10**-2.1,10**-0.75,'β = %0.2f [min]' % ((1./np.exp(params[4]))/60.), fontsize=font_size) 
plt.text(10**-2.51,10**-1.,r'FWHM = %0.2f [min]' % fwhm, fontsize=font_size)
#plt.savefig('C:/Users/Brendan/Desktop/spectra_temporal_averaging_methods_point.pdf', format='pdf', bbox_inches='tight')






try:                                 
    #M2_low = [0., 0.3, -0.01, 0.00001, -6.5, 0.05]
    #M2_high = [0.002, 6., 0.01, 0.2, -4.6, 0.8]
    M2_low = [0., 0.3, 0., 0.00001, 1., 0.]  # 1600
    M2_high = [0.0002, 3., 0.001, 0.2, 100., 0.1]
    
    nlfit_gp, nlpcov_gp = scipy.optimize.curve_fit(KappaPowerBase, f, s, bounds=(M2_low, M2_high), sigma=ds, method='dogbox', max_nfev=3000)
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
    
    nlfit_gp2, nlpcov_gp2 = scipy.optimize.curve_fit(KappaPowerBase, f, s, p0 = [A2, n2, C2, P2, fp2, fw2], bounds=(M2_low, M2_high), sigma=ds, max_nfev=3000)
    #nlfit_gp2, nlpcov_gp2 = scipy.optimize.curve_fit(LorentzPowerBase, f, s, bounds=(M2_low, M2_high), sigma=ds, max_nfev=3000)
   
except RuntimeError:
    #print("Error M2 - curve_fit failed - %i, %i" % (l,m))  # turn off because would print too many to terminal
    pass

except ValueError:
    #print("Error M2 - inf/NaN - %i, %i" % (l,m))  # turn off because would print too many to terminal
    pass

A22, n22, C22, P22, fp22, fw22 = nlfit_gp2  # unpack fitting parameters     
               
# create model functions from fitted parameters  
m1_fit = PowerLaw(f, A22, n22, C22)      
kappa = Kappa(f, P22,fp22,fw22) 
m2_fit2 = KappaPowerBase(f, A22,n22,C22,P22,fp22,fw22)      

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
    
ax3 = plt.subplot2grid((10,32),(0, 22), colspan=10, rowspan=10)
ax3.set_title('Kappa Function', fontsize=font_size)
plt.loglog(freqs, s, 'k')
plt.loglog(freqs, fit, color='purple')
plt.loglog(freqs, m1_fit, color='green', linestyle='dashed')
plt.loglog(freqs, kappa, color='green', linestyle='dashed')
plt.xlim(10**-4.5, 10**-1.)
plt.ylim(10**-4.5, 10**0) 
plt.text(10**-2.1,10**-0.5,'n = %0.2f' % params[1], fontsize=font_size)
plt.text(10**-2.1,10**-0.75,'κ = %0.2f' % params[4], fontsize=font_size) 
plt.text(10**-2.1,10**-1.,'ρ = %0.2e' % params[5], fontsize=font_size)
#plt.savefig('C:/Users/Brendan/Desktop/gauss_lorentz_kappa_%ix_%iy.pdf' % (x0[k],y0[k]), format='pdf', bbox_inches='tight')