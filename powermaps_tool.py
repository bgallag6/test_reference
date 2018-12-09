# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 10:14:10 2018

@author: Brendan
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RectangleSelector
import matplotlib.patches as patches
import scipy.signal
from pylab import axvline
import sunpy
import sunpy.cm
from scipy import fftpack
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.widgets import Button, Slider
from scipy.stats import f as ff
from scipy.stats.stats import pearsonr
import os

    
def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]

def plotMap(p):
    global cbar1
    global im
    cbar1.remove()
    im.remove()
    param = h_map[p]
    pflat = np.reshape(param, (param.shape[0]*param.shape[1]))
    pNaN = pflat[~np.isnan(pflat)]
    h_min = np.percentile(pNaN,1)  # set heatmap vmin to 1% of data (could lower to 0.5% or 0.1%)
    h_max = np.percentile(pNaN,99)  # set heatmap vmax to 99% of data (could up to 99.5% or 99.9%)
    if p == 4:
        c_map = 'jet_r'
    else:
        c_map = 'jet'
    im = ax1.imshow(param, cmap=c_map, interpolation='nearest', vmin=h_min, vmax=h_max, picker=True)
    ax1.set_title(r'%s: %i $\AA$ | %s' % (date_title, wavelength, titles[p]), y = 1.01, fontsize=17)
    colorBar()

def colorBar():
    global cax1
    global cbar1
    # design colorbar for heatmaps
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="3%", pad=0.07)
    cbar1 = plt.colorbar(im,cax=cax1)
    #cbar.set_label('%s' % cbar_labels[1], size=15, labelpad=10)
    cbar1.ax.tick_params(labelsize=13, pad=3)   
    plt.colorbar(im,cax=cax1)
    plt.draw()
    
def colorBar2():
    global cax2
    global cbar2
    # design colorbar for heatmaps
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="3%", pad=0.07)
    cbar2 = plt.colorbar(im2,cax=cax2)
    #cbar.set_label('%s' % cbar_labels[1], size=15, labelpad=10)
    cbar2.ax.tick_params(labelsize=13, pad=3)   
    plt.colorbar(im2,cax=cax2)
    plt.draw()
    
def update(val):
    global mask_val
    global im2
    cbar2.remove()
    im2.remove()
    #mask_val = np.log(slid_mask.val)
    #slid_mask.valtext.set_text(mask_val)
    mask_val = slid_mask.val
    mask_val = 1./(mask_val*60)
    ax2.clear()
    ax2.set_xlim(0, h_map.shape[2]-1)
    ax2.set_ylim(0, h_map.shape[1]-1) 
    
    idx = (np.abs(f_fit - mask_val)).argmin()
    
    #param = np.zeros((spectra.shape[0], spectra.shape[1]))
    param = np.copy(spectra[:,:,idx])
    
    pflat = np.reshape(param, (param.shape[0]*param.shape[1]))
    pNaN = pflat[~np.isnan(pflat)]
    h_min = np.percentile(pNaN,1)  # set heatmap vmin to 1% of data (could lower to 0.5% or 0.1%)
    h_max = np.percentile(pNaN,99)  # set heatmap vmax to 99% of data (could up to 99.5% or 99.9%)
    
    #ax1.set_title(r'%s: %i $\AA$ | %s | $f_{masked}$ = %0.1f%s' % (date_title, wavelength, titles[p], mask_percent, '%'), y = 1.01, fontsize=17)
    im2 = ax2.imshow(param, cmap='Greys', interpolation='nearest', vmin=h_min, vmax=h_max)
    colorBar2()
    #plt.colorbar(im2,cax=cax2)
    #plt.draw()
    return mask_val
    
class Index(object):
    ind = 0
         
    def coeff(self, event):
        global marker
        marker = 0
        plotMap(marker)  # could just do this
        return marker       

    def index(self, event):
        global marker
        marker = 1
        plotMap(marker)
        return marker     
        
    def roll(self, event):  # meh, should probably fix this
        global marker
        global cbar1
        global im
        cbar1.remove()
        im.remove()
        marker = 2
        paramA = h_map[0]
        paramn = h_map[1]
        paramC = h_map[2]
        param = (paramC/paramA)**(1./paramn)
        param = np.nan_to_num(param)/60.
        h_min = np.percentile(param,1)  # set heatmap vmin to 1% of data (could lower to 0.5% or 0.1%)
        h_max = np.percentile(param,99)  # set heatmap vmax to 99% of data (could up to 99.5% or 99.9%)
        im = ax1.imshow(param, cmap='jet', interpolation='nearest', vmin=h_min, vmax=h_max,  picker=True)
        ax1.set_title(r'%s: %i $\AA$ | %s' % (date_title, wavelength, titles[2]), y = 1.01, fontsize=17)
        colorBar()      
        
    def lorentz_amp(self, event):
        global marker
        global cbar1
        global im
        cbar1.remove()
        im.remove()
        marker = 3
        param = h_map[3]
        pflat = np.reshape(param, (param.shape[0]*param.shape[1]))
        pNaN = pflat[~np.isnan(pflat)]
        h_min = np.percentile(pNaN,1)  # set heatmap vmin to 1% of data (could lower to 0.5% or 0.1%)
        h_max = np.percentile(pNaN,99)  # set heatmap vmax to 99% of data (could up to 99.5% or 99.9%)
        im = ax1.imshow(param, cmap='jet', interpolation='nearest', vmin=h_min, vmax=h_max,  picker=True)
        ax1.set_title(r'%s: %i $\AA$ | %s' % (date_title, wavelength, titles[3]), y = 1.01, fontsize=17)
        colorBar()
    
    def lorentz_loc(self, event):
        global marker
        global cbar1
        global im
        cbar1.remove()
        im.remove()
        marker = 4
        param = h_map[4]
        pflat = np.reshape(param, (param.shape[0]*param.shape[1]))
        pNaN = pflat[~np.isnan(pflat)]
        h_min = np.percentile(pNaN,1)  # set heatmap vmin to 1% of data (could lower to 0.5% or 0.1%)
        h_max = np.percentile(pNaN,99)  # set heatmap vmax to 99% of data (could up to 99.5% or 99.9%)
        im = ax1.imshow(param, cmap='jet_r', interpolation='nearest', vmin=h_min, vmax=h_max,  picker=True)
        ax1.set_title(r'%s: %i $\AA$ | %s' % (date_title, wavelength, titles[4]), y = 1.01, fontsize=17)
        colorBar()
        
    def lorentz_wid(self, event):
        global marker
        global cbar1
        global im
        cbar1.remove()
        im.remove()
        marker = 5
        param = h_map[5]
        pflat = np.reshape(param, (param.shape[0]*param.shape[1]))
        pNaN = pflat[~np.isnan(pflat)]
        h_min = np.percentile(pNaN,1)  # set heatmap vmin to 1% of data (could lower to 0.5% or 0.1%)
        h_max = np.percentile(pNaN,99)  # set heatmap vmax to 99% of data (could up to 99.5% or 99.9%)
        im = ax1.imshow(param, cmap='jet', interpolation='nearest', vmin=h_min, vmax=h_max,  picker=True)
        ax1.set_title(r'%s: %i $\AA$ | %s' % (date_title, wavelength, titles[5]), y = 1.01, fontsize=17)
        colorBar()
        
    def fstat(self, event):
        global marker
        global cbar1
        global im
        cbar1.remove()
        im.remove()
        marker = 6
        param = h_map[6]
        pflat = np.reshape(param, (param.shape[0]*param.shape[1]))
        pNaN = pflat[~np.isnan(pflat)]
        h_min = np.percentile(pNaN,1)  # set heatmap vmin to 1% of data (could lower to 0.5% or 0.1%)
        h_max = np.percentile(pNaN,99)  # set heatmap vmax to 99% of data (could up to 99.5% or 99.9%)
        im = ax1.imshow(param, cmap='jet', interpolation='none', vmin=h_min, vmax=h_max,  picker=True)
        ax1.set_title(r'%s: %i $\AA$ | %s' % (date_title, wavelength, titles[6]), y = 1.01, fontsize=17)
        colorBar()
        
    def visual(self, event):
        global cbar1
        global im
        cbar1.remove()
        im.remove()
        param = vis
        h_min = np.percentile(param,1)  # set heatmap vmin to 1% of data (could lower to 0.5% or 0.1%)
        h_max = np.percentile(param,99)  # set heatmap vmax to 99% of data (could up to 99.5% or 99.9%)
        im = ax1.imshow(param, cmap='sdoaia%i' % wavelength, interpolation='nearest', vmin=h_min, vmax=h_max, picker=True)
        ax1.set_title(r'%s: %i $\AA$ | %s' % (date_title, wavelength, titles[7]), y = 1.01, fontsize=17)
        colorBar()
        
    def saveFig(self, event):
        global count

        outdir = 'C:/Users/Brendan/Desktop/Tool_Figures'

        if not os.path.exists(os.path.dirname(outdir)):
            try:
                print("Specified directory not found.")
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST: raise
        else:
            plt.savefig('%s/%s_%i_%i.pdf' % (outdir, date, wavelength, count), bbox_inches='tight')
        count += 1
        return count
  
    

"""
##############################################################################
##############################################################################
"""

plt.rcParams["font.family"] = "Times New Roman"
font_size = 20

directory = 'S:'
date = '20161012'
wavelength = 1600

#directory = 'F:'
#date = '20130626'
#wavelength = 1700

global spectra

cube_shape = np.load('%s/DATA/%s/%i/spectra_mmap_shape.npy' % (directory, date, wavelength))
spectra = np.memmap('%s/DATA/%s/%i/spectra_mmap.npy' % (directory, date, wavelength), dtype='float64', mode='r', shape=(cube_shape[0], cube_shape[1], cube_shape[2]))

h_map = np.load('%s/DATA/%s/%i/param.npy' % (directory, date, wavelength))
vis = np.load('%s/DATA/%s/%i/visual.npy' % (directory, date, wavelength))

h_map[4] = (1./(np.exp(h_map[4]))/60.)

global marker
global count
marker = 1
count = 0

### determine frequency values that FFT will evaluate
global f_fit
if wavelength == 1600 or wavelength == 1700:
    time_step = 24
else:
    time_step = 12
freq_size = (cube_shape[2]*2)+1
sample_freq = fftpack.fftfreq(freq_size, d=time_step)
pidxs = np.where(sample_freq > 0)    
freqs = sample_freq[pidxs]
#print(len(freqs))
f_fit = np.linspace(freqs[0],freqs[len(freqs)-1],int(spectra.shape[2])) 

# create list of titles and colorbar names for display on the figures
titles = ['Power Law Slope Coeff.', 'Power Law Index', 'Rollover [min]', 'Lorentzian Amplitude', 'Lorentzian Location [min]', 'Lorentzian Width', 'F-Statistic', 'Averaged Visual Image']
date_title = '%i/%02i/%02i' % (int(date[0:4]),int(date[4:6]),int(date[6:8]))

# create figure with heatmap and spectra side-by-side subplots
fig1 = plt.figure(figsize=(20,10))

ax1 = plt.gca()
ax1 = plt.subplot2grid((31,30),(4, 1), colspan=14, rowspan=25)
ax1.set_xlim(0, h_map.shape[2]-1)
ax1.set_ylim(0, h_map.shape[1]-1)  
ax1.set_title(r'%s: %i $\AA$ | %s' % (date_title, wavelength, titles[1]), y = 1.01, fontsize=17)

# was getting error "'AxesImage' object is not iterable"
# "Each element in img needs to be a sequence of artists, not a single artist."
param = h_map[1]  # set initial heatmap to power law index     
h_min = np.percentile(param,1)  # set heatmap vmin to 1% of data (could lower to 0.5% or 0.1%)
h_max = np.percentile(param,99)  # set heatmap vmax to 99% of data (could up to 99.5% or 99.9%)
im, = ([ax1.imshow(param, cmap='jet', interpolation='nearest', vmin=h_min, vmax=h_max,  picker=True)])

# design colorbar for heatmaps
global cbar1
divider1 = make_axes_locatable(ax1)
cax1 = divider1.append_axes("right", size="3%", pad=0.07)
cbar1 = plt.colorbar(im,cax=cax1)
#cbar.set_label('%s' % cbar_labels[1], size=15, labelpad=10)
cbar1.ax.tick_params(labelsize=13, pad=3)   


# make toggle buttons to display each parameter's heatmap
axcoeff = plt.axes([0.01, 0.9, 0.05, 0.063])
axindex = plt.axes([0.07, 0.9, 0.05, 0.063])
axroll = plt.axes([0.13, 0.9, 0.05, 0.063])
axlorentz_amp = plt.axes([0.19, 0.9, 0.05, 0.063])
axlorentz_loc = plt.axes([0.25, 0.9, 0.05, 0.063])
axlorentz_wid = plt.axes([0.31, 0.9, 0.05, 0.063])
axfstat = plt.axes([0.37, 0.9, 0.05, 0.063])
axvisual = plt.axes([0.43, 0.9, 0.05, 0.063])
axslider = plt.axes([0.58, 0.915, 0.3, 0.04])
axsaveFig = plt.axes([0.92, 0.9, 0.05, 0.063])



# set up spectra subplot
ax2 = plt.subplot2grid((31,30),(4, 16), colspan=14, rowspan=25)
ax2.set_xlim(0, h_map.shape[2]-1)
ax2.set_ylim(0, h_map.shape[1]-1)  
ax1.set_title(r'%s: %i $\AA$ | %s' % (date_title, wavelength, titles[1]), y = 1.01, fontsize=17)

# was getting error "'AxesImage' object is not iterable"
# - found: "Each element in img needs to be a sequence of artists, not a single artist."
idx = (np.abs(f_fit - 1./(4*60))).argmin()
param = np.copy(spectra[:,:,idx])  # set initial heatmap to power law index     
pflat = np.reshape(param, (param.shape[0]*param.shape[1]))
pNaN = pflat[~np.isnan(pflat)]
h_min = np.percentile(pNaN,1)  # set heatmap vmin to 1% of data (could lower to 0.5% or 0.1%)
h_max = np.percentile(pNaN,99)  # set heatmap vmax to 99% of data (could up to 99.5% or 99.9%)
im2, = ([ax2.imshow(param, cmap='Greys', interpolation='nearest', vmin=h_min, vmax=h_max)])  

global cbar2
# design colorbar for heatmaps
divider2 = make_axes_locatable(ax2)
cax2 = divider2.append_axes("right", size="3%", pad=0.07)
cbar2 = plt.colorbar(im2,cax=cax2)
cbar2.ax.tick_params(labelsize=13, pad=3)   

plt.tight_layout()


# add callbacks to each button - linking corresponding action
callback = Index()

bcoeff = Button(axcoeff, 'Coeff.')
bcoeff.on_clicked(callback.coeff)
bindex = Button(axindex, 'Index')
bindex.on_clicked(callback.index)
broll = Button(axroll, 'Rollover')
broll.on_clicked(callback.roll)
blorentz_amp = Button(axlorentz_amp, 'Lorentz. Amp')
blorentz_amp.on_clicked(callback.lorentz_amp)
blorentz_loc = Button(axlorentz_loc, 'Lorentz. Loc')
blorentz_loc.on_clicked(callback.lorentz_loc)
blorentz_wid = Button(axlorentz_wid, 'Lorentz. Wid')
blorentz_wid.on_clicked(callback.lorentz_wid)
bfstat = Button(axfstat, 'F-Stat')
bfstat.on_clicked(callback.fstat)
bvisual = Button(axvisual, 'Visual')
bvisual.on_clicked(callback.visual)
bsaveFig = Button(axsaveFig, 'Save')
bsaveFig.on_clicked(callback.saveFig)

#slid_mask = Slider(axslider, 'Frequency', f_fit[0], f_fit[-1], valinit=(1./240))
#slid_mask = Slider(axslider, 'Period', (1./f_fit[-1])/60., (1./f_fit[0])/60., valinit=4., valfmt='%0.2f')
slid_mask = Slider(axslider, 'Period', (1./f_fit[-1])/60., 50., valinit=4., valfmt='%0.2f')
slid_mask.on_changed(update)