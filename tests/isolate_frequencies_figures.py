# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 10:39:12 2018

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
 

"""
directory = 'S:'
date = '20130626'
wavelength = 171
n_segments = 6
"""

directory = 'F:'
date = '20140818'
wavelength = 1600

font_size = 15

# determine frequency values that FFT will evaluate
if wavelength in [1600,1700]:
    time_step = 24  # add as argument in function call, or leave in as constant?
else:
    time_step = 12


# 1600 20140818
x0 = [153, 41, 22, 41, 84, 88, 94, 95, 98, 184, 138, 96, 107, 47]
y0 = [104, 152, 129, 45, 137, 83, 93, 70, 83, 152, 136, 81, 86, 135]

k = 0
 
cube_shape = np.load('%s/DATA/Temp/%s/%i/spectra_mmap_shape.npy' % (directory, date, wavelength))
spectra = np.memmap('%s/DATA/Temp/%s/%i/spectra_mmap.npy' % (directory, date, wavelength), dtype='float64', mode='r', shape=(cube_shape[0], cube_shape[1], cube_shape[2]))

sample_freq = fftpack.fftfreq(spectra.shape[2]*2+2, d=time_step)
pidxs = np.where(sample_freq > 0)
freqs = sample_freq[pidxs]

params = np.zeros((11))

for i in range(len(x0)):
#for i in range(1):
    s = spectra[y0[i]][x0[i]]

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
    
    fwhm = (1. / (np.exp(params[4]+params[5]) - np.exp(params[4]-params[5])))/60.
    
    plt.rcParams["font.family"] = "Times New Roman"
    font_size = 17
        
    plt.figure(figsize=(20,6))
    plt.suptitle('Pixel: %ix, %iy' % (x0[i], y0[i]), fontsize=font_size)
    ax1 = plt.subplot2grid((10,32),(0, 0), colspan=10, rowspan=10)
    ax1.set_title('Full Frequencies', fontsize=font_size)
    plt.loglog(f, s, 'k')
    plt.loglog(f, fit, color='purple')
    plt.loglog(f, m1_fit, color='green', linestyle='dashed')
    plt.loglog(f, lorentz, color='green', linestyle='dashed')
    plt.xlim(10**-4.3, 10**-1.5)
    plt.ylim(10**-4.5, 10**0) 
    plt.text(10**-2.35,10**-0.5,'n = %0.2f' % params[1], fontsize=font_size)
    plt.text(10**-2.35,10**-0.75,'β = %0.2f [min]' % ((1./np.exp(params[4]))/60.), fontsize=font_size) 
    plt.text(10**-2.71,10**-1.,r'FWHM = %0.2f [min]' % fwhm, fontsize=font_size)
    plt.vlines(1./300., 10**-4.5, 10**0, linestyle='--', color='k', linewidth=0.5)
    plt.vlines(1./180., 10**-4.5, 10**0, linestyle='--', color='k', linewidth=0.5)
    #plt.savefig('C:/Users/Brendan/Desktop/spectra_temporal_averaging_methods_point.pdf', format='pdf', bbox_inches='tight')
    
    
    
    
    
    
    k = 80
    s = s[:k]
    f = f[:k]
    ds = ds[:k]
    
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
        
    fwhm = (1. / (np.exp(params[4]+params[5]) - np.exp(params[4]-params[5])))/60.
        
    ax2 = plt.subplot2grid((10,32),(0, 11), colspan=10, rowspan=10)
    ax2.set_title('Freqs. > 1.5 Minutes', fontsize=font_size)
    plt.loglog(f, s, 'k')
    plt.loglog(f, fit, color='purple')
    plt.loglog(f, m1_fit, color='green', linestyle='dashed')
    plt.loglog(f, lorentz, color='green', linestyle='dashed')
    plt.xlim(10**-4.3, 10**-1.5)
    plt.ylim(10**-4.5, 10**0) 
    plt.text(10**-2.35,10**-0.5,'n = %0.2f' % params[1], fontsize=font_size)
    plt.text(10**-2.35,10**-0.75,'β = %0.2f [min]' % ((1./np.exp(params[4]))/60.), fontsize=font_size) 
    plt.text(10**-2.71,10**-1.,r'FWHM = %0.2f [min]' % fwhm, fontsize=font_size)
    plt.vlines(1./300., 10**-4.5, 10**0, linestyle='--', color='k', linewidth=0.5)
    plt.vlines(1./180., 10**-4.5, 10**0, linestyle='--', color='k', linewidth=0.5)
    #plt.savefig('C:/Users/Brendan/Desktop/spectra_temporal_averaging_methods_point.pdf', format='pdf', bbox_inches='tight')
    
    
    
    
    k1 = 19
    k2 = 60
    s = s[k1:k2]
    f = f[k1:k2]
    ds = ds[k1:k2]
    
    try:                                 
        #M2_low = [0., 0.3, -0.01, 0.00001, -6.5, 0.05]
        #M2_high = [0.002, 6., 0.01, 0.2, -4.6, 0.8]
        M2_low = [0.00001, -6.5, 0.05]
        M2_high = [0.2, -4.6, 0.8]
        
        nlfit_gp, nlpcov_gp = scipy.optimize.curve_fit(Lorentz, f, s, bounds=(M2_low, M2_high), sigma=ds, method='dogbox', max_nfev=3000)
        #nlfit_gp, nlpcov_gp = scipy.optimize.curve_fit(LorentzPowerBase, f, s, p0 = [A,n,C,0.1,-5.55,0.425], bounds=(M2_low, M2_high), sigma=ds, method='dogbox', max_nfev=3000)
    
    except RuntimeError:
        #print("Error M2 - curve_fit failed - %i, %i" % (l,m))  # turn off because would print too many to terminal
        pass
    
    except ValueError:
        #print("Error M2 - inf/NaN - %i, %i" % (l,m))  # turn off because would print too many to terminal
        pass
    
    P2, fp2, fw2 = nlfit_gp  # unpack fitting parameters
    
    # next fit using default 'trf' method
    try:
        
        nlfit_gp2, nlpcov_gp2 = scipy.optimize.curve_fit(Lorentz, f, s, p0 = [P2, fp2, fw2], bounds=(M2_low, M2_high), sigma=ds, max_nfev=3000)
        #nlfit_gp2, nlpcov_gp2 = scipy.optimize.curve_fit(LorentzPowerBase, f, s, bounds=(M2_low, M2_high), sigma=ds, max_nfev=3000)
       
    except RuntimeError:
        #print("Error M2 - curve_fit failed - %i, %i" % (l,m))  # turn off because would print too many to terminal
        pass
    
    except ValueError:
        #print("Error M2 - inf/NaN - %i, %i" % (l,m))  # turn off because would print too many to terminal
        pass
    
    P22, fp22, fw22 = nlfit_gp2  # unpack fitting parameters     
                   
    # create model functions from fitted parameters        
    lorentz = Lorentz(f, P22,fp22,fw22)     
     
    
    residsM22 = (s - lorentz)
    chisqrM22 = ((residsM22/ds)**2).sum()
    redchisqrM22 = chisqrM22 / float(f.size-6)         

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
    
    fit = lorentz
    
    fwhm = (1. / (np.exp(params[4]+params[5]) - np.exp(params[4]-params[5])))/60.
        
        
    ax3 = plt.subplot2grid((10,32),(0, 22), colspan=10, rowspan=10)
    ax3.set_title('2 Minutes < Freqs. < 6 Minutes', fontsize=font_size)
    plt.loglog(f, s, 'k')
    plt.loglog(f, fit, color='purple')
    plt.loglog(f, lorentz, color='green', linestyle='dashed')
    plt.xlim(10**-4.3, 10**-1.5)
    plt.ylim(10**-4.5, 10**0) 
    plt.text(10**-2.35,10**-0.75,'β = %0.2f [min]' % ((1./np.exp(params[4]))/60.), fontsize=font_size) 
    plt.text(10**-2.71,10**-1.,r'FWHM = %0.2f [min]' % fwhm, fontsize=font_size)
    plt.vlines(1./300., 10**-4.5, 10**0, linestyle='--', color='k', linewidth=0.5)
    plt.vlines(1./180., 10**-4.5, 10**0, linestyle='--', color='k', linewidth=0.5)
    
    plt.savefig('C:/Users/Brendan/Desktop/isolate_frequencies_%ix_%iy.pdf' % (x0[i],y0[i]), format='pdf', bbox_inches='tight')