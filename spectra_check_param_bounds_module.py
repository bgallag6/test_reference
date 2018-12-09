# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 14:37:47 2018

@author: Brendan
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
#matplotlib.use('TkAgg') 	# NOTE: This is a MAC/OSX thing. Probably REMOVE for linux/Win
from pylab import axvline
from scipy import fftpack
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.widgets import Button, RadioButtons, Slider
from scipy.stats import f as ff
import os
from sunpy.map import Map

    
def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]
    
def update(val):
    #global A2, n2, C2, P2, fp2, fw2
    global P0, P1, P2, P3, P4, P5
    
    P0 = 10**scoeff.val
    scoeff.valtext.set_text('%0.2e' % P0)
    P1 = sindex.val
    sindex.valtext.set_text('%0.2f' % P1)
    P2 = stail.val
    stail.valtext.set_text('%0.2e' % P2)
    P3 = samp.val
    samp.valtext.set_text('%0.2e' % P3)
    P4 = sloc.val
    sloc.valtext.set_text('%0.2f' % (1./(np.exp(P4)*60.)))
    P5 = swid.val
    swid.valtext.set_text('%0.2f' % P5)
    #l.set_ydata(amp*np.sin(2*np.pi*freq*t))
    params = P0, P1, P2, P3, P4, P5
       
    s = LorentzPowerBase(freqs, *params)
    
    l.set_ydata(s)


def reset(event):
    scoeff.reset()
    sindex.reset()
    stail.reset()
    samp.reset()
    sloc.reset()
    swid.reset()
    m2.set_ydata(emptyLine)
    l2.set_ydata(emptyLine)
    t1.set_text('')
    t2.set_text('')
    t3.set_text('')
    t4.set_text('')

    
class Index(object):
    ind = 0
    
    def specFit(self, event):
        #global A2, n2, C2, P2, fp2, fw2
        global P0, P1, P2, P3, P4, P5
        s = np.zeros((spectra.shape[2]))
    
        s[:] = spectra[iy][ix][:]
        
        # assign equal weights to all parts of the curve
        df = np.log10(freqs[1:len(freqs)]) - np.log10(freqs[0:len(freqs)-1])
        df2 = np.zeros_like(freqs)
        df2[0:len(df)] = df
        df2[len(df2)-1] = df2[len(df2)-2]
        ds = df2
        #ds = subcube_StdDev[l][m]
        
                                               
        ### fit data to models using SciPy's Levenberg-Marquart method                             
        ## fit data to combined power law plus gaussian component model
        
        try:
            # initial guesses for fitting parameters
            M1_low = [-0.002, 0.3, -0.01]
            M1_high = [0.002, 6., 0.01]
            m1Param = scipy.optimize.curve_fit(PowerLaw, freqs, s, bounds=(M1_low, M1_high), sigma=ds, method='dogbox')[0]
          
        except RuntimeError: pass
        
        except ValueError: pass
        
        A, n, C = m1Param  # unpack fitting parameters
         
        """        
        try:                                 
            M2_low = [0., 0.3, -0.01, 0.00001, -6.5, 0.05]
            M2_high = [0.002, 6., 0.01, 0.2, -4.6, 0.8]
                       
            # change method to 'dogbox' and increase max number of function evaluations to 3000
            nlfit_gp, nlpcov_gp = scipy.optimize.curve_fit(LorentzPowerBase, freqs, s, bounds=(M2_low, M2_high), sigma=ds, method='dogbox', max_nfev=3000)
        
        except RuntimeError: pass
        
        except ValueError: pass
        
        
        A2, n2, C2, P2, fp2, fw2 = nlfit_gp  # unpack fitting parameters
        #print nlfit_gp
        #"""
               
        try:
            m2Param = scipy.optimize.curve_fit(LorentzPowerBase, freqs, s, p0 = [P0, P1, P2, P3, P4, P5], sigma=ds)[0]
            
           
        except RuntimeError: pass 
        except ValueError: pass
        
        A22, n22, C22, P22, fp22, fw22 = m2Param  # unpack fitting parameters     
        params = m2Param
                       
        # create model functions from fitted parameters    
        m1_fit = PowerLaw(freqs, A, n, C)    
        lorentz = Lorentz(freqs,P22,fp22,fw22)
        m2_fit2 = LorentzPowerBase(freqs, A22,n22,C22,P22,fp22,fw22)      
        
        residsM1 = (s - m1_fit)
        chisqrM1 =  ((residsM1/ds)**2).sum()
        #redchisqrM1 = ((residsM1/ds)**2).sum()/float(freqs.size-3)  
        
        residsM22 = (s - m2_fit2)
        chisqrM22 = ((residsM22/ds)**2).sum()
        redchisqrM22 = ((residsM22/ds)**2).sum()/float(freqs.size-6) 
              
        #f_test2 = ((chisqrM1-chisqrM22)/(6-3))/((chisqrM22)/(freqs.size-6))
        
        #df1, df2 = 3, 6  # degrees of freedom for model M1, M2
        #p_val = ff.sf(f_test2, df1, df2)
    
        #amp_scale2 = PowerLaw(np.exp(fp22), A22, n22, C22)  # to extract the gaussian-amplitude scaling factor   
        
        #rollover = (1. / ((C22 / A22)**(-1. / n22))) / 60.
        
        fwhm = (1. / (np.exp(params[4]+params[5]) - np.exp(params[4]-params[5])))/60.
        
        #ax2.loglog(f, m1_fit, 'r', linewidth=1.3, label='M1')
        #ax2.loglog(f, m2_fit, 'b', linewidth=1.3, label='M2')
        m2.set_ydata(m2_fit2)
        l2.set_ydata(lorentz)

        t1.set_text(r'$n$ = %0.2f' % params[1])
        t2.set_text(r'$\beta$ = %0.2f [min]' % ((1./np.exp(params[4]))/60.))
        t3.set_text(r'FWHM = %0.2f [min]' % fwhm)
        t4.set_text(r'$\chi_\nu^2$ = {0:0.2f}'.format(redchisqrM22))
        #plt.vlines((0.0093),10**-8,10**1, linestyles='dotted', label='3 minutes')
        legend = ax2.legend(loc='lower left', prop={'size':15}, labelspacing=0.35)   
        for label in legend.get_lines():
                label.set_linewidth(2.0)  # the legend line width 
                
        
    def saveFig(self, event):
        global count

        outdir = 'C:/Users/Brendan/Desktop/Tool_Figures/'

        if not os.path.exists(os.path.dirname(outdir)):
            try:
                print("Specified directory not found.")
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST: raise
        else:
            plt.savefig('%s%s_%i_%i.pdf' % (outdir, date, wavelength, count), bbox_inches='tight')
        count += 1
        return count
  

# Simple mouse click function to store coordinates
def onclick(event):
    global ix, iy, c
    ixx, iyy = event.xdata, event.ydata
    if ixx > 1. and iyy > 1.:
        del ax1.collections[:]
        plt.draw()
        
        print ('x = %d, y = %d' % ( ixx, iyy))  # print location of pixel
        ix = int(ixx)
        iy = int(iyy)
        
        t1.set_text('')
        t2.set_text('')
        t3.set_text('')
        t4.set_text('')
        m2.set_ydata(emptyLine)
        l2.set_ydata(emptyLine)
        
        s = np.zeros((spectra.shape[2]))
    
        s[:] = spectra[iy][ix][:]
        l0.set_ydata(s)
        
        ax1.scatter(ix, iy, s=200, marker='x', c='white', linewidth=2.5)
        
    return ix, iy
    
# define combined-fitting function (Model M2)
def LorentzPowerBase(f2, A2, n2, C2, P2, fp2, fw2):
    return A2*f2**-n2 + C2 + P2*(1./ ((np.pi*fw2)*(1.+((np.log(f2)-fp2)/fw2)**2)))
             
# define combined-fitting function (Model M2)
def GaussPowerBase(f2, A2, n2, C2, P2, fp2, fw2):
    return A2*f2**-n2 + C2 + P2*np.exp(-0.5*(((np.log(f2))-fp2)/fw2)**2)    

# define Power-Law-fitting function (Model M1)
def PowerLaw(f, A, n, C):
    return A*f**-n + C
        
# define Gaussian-fitting function
def Lorentz(f, P, fp, fw):
    return P*(1./ ((np.pi*fw)*(1.+((np.log(f)-fp)/fw)**2)))
    
# define Gaussian-fitting function
def Gauss(f, P, fp, fw):
    return P*np.exp(-0.5*(((np.log(f))-fp)/fw)**2) 
    

"""
##############################################################################
##############################################################################
"""

directory = 'S:'
date = '20120606'
wavelength = 1600

global spectra
global freqs
global count
global emptyLine

cube_shape = np.load('%s/DATA/%s/%i/spectra_mmap_shape.npy' % (directory, date, wavelength))
spectra = np.memmap('%s/DATA/%s/%i/spectra_mmap.npy' % (directory, date, wavelength), dtype='float64', mode='r', shape=(cube_shape[0], cube_shape[1], cube_shape[2]))

count = 0

### determine frequency values that FFT will evaluate
if wavelength == 1600 or wavelength == 1700:
    time_step = 24
else:
    time_step = 12
freq_size = (cube_shape[2]*2)+2
sample_freq = fftpack.fftfreq(freq_size, d=time_step)
pidxs = np.where(sample_freq > 0)    
freqs = sample_freq[pidxs] 

font_size = 15


#global A2, n2, C2, P2, fp2, fw2
global P0, P1, P2, P3, P4, P5

P0 = 1e-5
P1 = 1.
P2 = 1e-4
P3 = 0.01
P4 = -5.48
P5 = 0.1
params = P0, P1, P2, P3, P4, P5


 
vis = np.load('%s/DATA/%s/%i/visual.npy' % (directory, date, wavelength))

if len(vis.shape) > 1:
    vis = vis[0]
    
h_min = np.percentile(vis,1)
h_max = np.percentile(vis,99)


date_title = '%i/%02i/%02i' % (int(date[0:4]),int(date[4:6]),int(date[6:8]))

#s = LorentzPowerBase(freqs,A0,n0,C0,P0,fp0,fw0)
s = LorentzPowerBase(freqs, *params)

# create figure with heatmap and spectra side-by-side subplots
fig1 = plt.figure(figsize=(16,9))
plt.subplots_adjust(bottom=0.4)

ax1 = plt.gca()
ax1 = plt.subplot2grid((30,31),(0, 1), colspan=14, rowspan=20)

ax1.set_xlim(0, vis.shape[1]-1)
ax1.set_ylim(0, vis.shape[0]-1)  
ax1.set_title(r'%i $\AA$ | Averaged Visual Image' % wavelength, y = 1.01, fontsize=17)


im = ax1.imshow(vis, cmap='sdoaia%i' % wavelength, interpolation='nearest', vmin=h_min, vmax=h_max, picker=True)

# design colorbar for heatmaps
divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="3%", pad=0.07)
cbar = plt.colorbar(im,cax=cax)
cbar.ax.tick_params(labelsize=13, pad=3)   

# make toggle buttons to display each parameter's heatmap
axcoeff = plt.axes([0.15, 0.23, 0.6, 0.02])
axindex = plt.axes([0.15, 0.19, 0.6, 0.02])
axtail = plt.axes([0.15, 0.15, 0.6, 0.02])
axamp = plt.axes([0.15, 0.11, 0.6, 0.02])
axloc = plt.axes([0.15, 0.07, 0.6, 0.02])
axwid = plt.axes([0.15, 0.03, 0.6, 0.02])
axspecFit = plt.axes([0.83, 0.18, 0.05, 0.05])
axsaveFig = plt.axes([0.83, 0.11, 0.05, 0.05])
axreset = plt.axes([0.83, 0.04, 0.05, 0.05])

emptyLine = [0 for i in range(len(freqs))]

global ix, iy
ix = spectra.shape[1]//2
iy = spectra.shape[0]//2

# set up spectra subplot
ax2 = plt.subplot2grid((30,31),(0, 16), colspan=14, rowspan=20)
ax2.set_title('Spectra Fit', y = 1.01, fontsize=17)
l0, = ax2.loglog(freqs, spectra[iy][ix], 'k')
l, = ax2.loglog(freqs, s, lw=1.5, color='red')
m2, = ax2.loglog(freqs, emptyLine, 'b', linewidth=1.3, label='M2 - Lorentz')
l2, = ax2.loglog(freqs, emptyLine, 'b--', linewidth=1.3, label='Lorentz')
ax2.set_xlim(10**-4.5, 10**-1.3)
ax2.set_ylim(10**-5, 10**0)  
ax2.set_xlabel('Frequency [Hz]', fontsize=font_size, labelpad=2)
ax2.set_ylabel('Power', fontsize=font_size, labelpad=2)
t1, = ([plt.text(10**-2.35, 10**-0.5, '', fontsize=font_size)])
t2, = ([plt.text(10**-2.35, 10**-0.75, '', fontsize=font_size)])
t3, = ([plt.text(10**-2.60, 10**-1., '', fontsize=font_size)])
t4, = ([plt.text(10**-2.40, 10**-1.25, '', fontsize=font_size)])

fig1.canvas.mpl_connect('button_press_event', onclick)


plt.tight_layout()

param_names = ['Coeff.', 'Index', 'Tail', 'Amp.', 'Loc.', 'Wid.']

# add callbacks to each button - linking corresponding action
callback = Index()

bspecFit = Button(axspecFit, 'specFit')
bspecFit.on_clicked(callback.specFit)    
bsaveFig = Button(axsaveFig, 'Save')
bsaveFig.on_clicked(callback.saveFig)
breset = Button(axreset, 'Reset')
breset.on_clicked(reset)
scoeff = Slider(axcoeff, '%s' % param_names[0], -15, -3, valinit=np.log10(P0))
scoeff.on_changed(update)
sindex = Slider(axindex, '%s' % param_names[1], 0.3, 4.0, valinit=P1)
sindex.on_changed(update)
stail = Slider(axtail, '%s' % param_names[2], -0.01, 0.03, valinit=P2)
stail.on_changed(update)
samp = Slider(axamp, '%s' % param_names[3], 1e-7, 0.1, valinit=P3)
samp.on_changed(update)
sloc = Slider(axloc, '%s' % param_names[4], -6.4, -4.6, valinit=P4)
sloc.on_changed(update)
swid = Slider(axwid, '%s' % param_names[5], 0.05, 0.8, valinit=P5)
swid.on_changed(update)

    