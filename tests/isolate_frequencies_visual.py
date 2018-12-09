# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 11:21:13 2018

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
x0 = [153, 22, 41, 84, 88, 95, 98, 184, 96, 107]
y0 = [104, 129, 45, 137, 83, 70, 83, 152, 81, 86]

k = 0
 
vis = np.load('%s/DATA/Output/%s/%i/visual.npy' % (directory, date, wavelength))
vis0 = vis[0, 1:-1, 1:-1]

vflat = np.reshape(vis0, (vis0.shape[0]*vis0.shape[1]))
v_min = np.percentile(vflat, 1)
v_max = np.percentile(vflat, 99)

plt.figure(figsize=(12,8))
ax = plt.gca()
ax.set_title('2014/08/18 1600 $\AA$', y = 1.01, fontsize=17)
ax.imshow(vis0, cmap='sdoaia1600', vmin=v_min, vmax=v_max)
ax.scatter(x0, y0, s=40, c='white')
ax.scatter(x0, y0, s=30, c='red')
ax.scatter(x0, y0, s=15, c='white')
ax.set_xlim(0, vis0.shape[1]-1)
ax.set_ylim(0, vis0.shape[0]-1) 
#plt.savefig('C:/Users/Brendan/Desktop/20140818_1600_isolate_freqs_visual.pdf', format='pdf', bbox_inches='tight')
    