# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 18:01:07 2018

@author: Brendan
"""

import matplotlib.pyplot as plt
import numpy as np
import sunpy
import sunpy.cm
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.widgets import Button, Slider

    
def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]

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
    ax2.set_xlim(0, spectra.shape[1]-1)
    ax2.set_ylim(0, spectra.shape[0]-1) 
    
    idx = (np.abs(f_fit - mask_val)).argmin()
    
    #param = np.zeros((spectra.shape[0], spectra.shape[1]))
    param = np.copy(spectra[:,:,idx])
    
    pflat = np.reshape(param, (param.shape[0]*param.shape[1]))
    pNaN = pflat[~np.isnan(pflat)]
    h_min = np.percentile(pNaN,1)  # set heatmap vmin to 1% of data (could lower to 0.5% or 0.1%)
    h_max = np.percentile(pNaN,99)  # set heatmap vmax to 99% of data (could up to 99.5% or 99.9%)
    
    #ax1.set_title(r'%s: %i $\AA$ | %s | $f_{masked}$ = %0.1f%s' % (date_title, wavelength, titles[p], mask_percent, '%'), y = 1.01, fontsize=17)
    im2 = ax2.imshow(param, cmap=c_map, interpolation='nearest', vmin=h_min, vmax=h_max)
    #ax2.set_title(r'Phase Heatmap [deg] @ Period = %0.2 minutes' % (1./freqs[idx]), y = 1.01, fontsize=17)
    ax2.set_title(r'Phase [deg] @ Period = %0.2f minutes' % ((1./freqs[idx])/60), y = 1.01, fontsize=17)
    colorBar2()
    #plt.colorbar(im2,cax=cax2)
    #plt.draw()
    return mask_val
  
    

"""
##############################################################################
##############################################################################
"""
import matplotlib.colors as col
import seaborn as sns
import hsluv # install via pip

##### generate custom colormaps
def make_segmented_cmap(): 
    white = '#ffffff'
    black = '#000000'
    red = '#ff0000'
    blue = '#0000ff'
    anglemap = col.LinearSegmentedColormap.from_list(
        'anglemap', [black, red, white, blue, black], N=256, gamma=1)
    return anglemap

def make_anglemap( N = 256, use_hpl = True ):
    h = np.ones(N) # hue
    h[:N//2] = 11.6 # red 
    h[N//2:] = 258.6 # blue
    s = 100 # saturation
    l = np.linspace(0, 100, N//2) # luminosity
    l = np.hstack( (l,l[::-1] ) )

    colorlist = np.zeros((N,3))
    for ii in range(N):
        if use_hpl:
            colorlist[ii,:] = hsluv.hpluv_to_rgb( (h[ii], s, l[ii]) )
        else:
            colorlist[ii,:] = hsluv.hsluv_to_rgb( (h[ii], s, l[ii]) )
    colorlist[colorlist > 1] = 1 # correct numeric errors
    colorlist[colorlist < 0] = 0 
    return col.ListedColormap( colorlist )

N = 256
segmented_cmap = make_segmented_cmap()
flat_huslmap = col.ListedColormap(sns.color_palette('husl',N))
hsluv_anglemap = make_anglemap( use_hpl = False )
hpluv_anglemap = make_anglemap( use_hpl = True )
colormaps = [segmented_cmap, flat_huslmap, hsluv_anglemap, hpluv_anglemap]

c_map = colormaps[3]

#directory = "C:/Users/Brendan/Desktop/specFit/test/Processed/20120606/1600"
directory = "C:/Users/Brendan/Desktop/specFit/images/processed/20120606/1600"
#directory = "C:/Users/Brendan/Desktop/from_specfit_test_processed_1600"

#phaseArr = np.load('%s/spectraPhase.npy' % directory)
phaseArr = np.load('%s/spectraPhaseB.npy' % directory)
#phaseArr = np.load('%s/phaseCube.npy' % directory)
#specArr = np.load('%s/specCube.npy' % directory)
freqs = np.load('%s/frequencies.npy' % directory)

"""
for i in range(20,35):
    plt.figure()
    plt.title('Period: %0.2f' % (1./(freqs[i]*60)))
    plt.imshow(phaseArr[:,:,i], cmap='jet')
"""    
    
plt.rcParams["font.family"] = "Times New Roman"
font_size = 20

global spectra

spectra = phaseArr

vis = np.load('%s/visual.npy' % directory)

f_fit = freqs
date = '20120606'
wavelength = 1600

# create list of titles and colorbar names for display on the figures
titles = ['Power Law Slope Coeff.', 'Power Law Index', 'Rollover [min]', 'Lorentzian Amplitude', 'Lorentzian Location [min]', 'Lorentzian Width', 'F-Statistic', 'Averaged Visual Image']
date_title = '%i/%02i/%02i' % (int(date[0:4]),int(date[4:6]),int(date[6:8]))

# create figure with heatmap and spectra side-by-side subplots
fig1 = plt.figure(figsize=(20,10))

ax1 = plt.gca()
ax1 = plt.subplot2grid((31,30),(4, 1), colspan=14, rowspan=25)
ax1.set_xlim(0, vis.shape[1]-1)
ax1.set_ylim(0, vis.shape[0]-1)  
ax1.set_title(r'%s: %i $\AA$ | Visual Image' % (date_title, wavelength), y = 1.01, fontsize=17)

# was getting error "'AxesImage' object is not iterable"
# "Each element in img needs to be a sequence of artists, not a single artist."
param = vis  # set initial heatmap to power law index     
h_min = np.percentile(param,1)  # set heatmap vmin to 1% of data (could lower to 0.5% or 0.1%)
h_max = np.percentile(param,99)  # set heatmap vmax to 99% of data (could up to 99.5% or 99.9%)
im, = ([ax1.imshow(param, cmap='sdoaia1600', interpolation='nearest', vmin=h_min, vmax=h_max,  picker=True)])

# design colorbar for heatmaps
global cbar1
divider1 = make_axes_locatable(ax1)
cax1 = divider1.append_axes("right", size="3%", pad=0.07)
cbar1 = plt.colorbar(im,cax=cax1)
cbar1.ax.tick_params(labelsize=13, pad=3)   



# set up spectra subplot
ax2 = plt.subplot2grid((31,30),(4, 16), colspan=14, rowspan=25)
ax2.set_xlim(0, spectra.shape[1]-1)
ax2.set_ylim(0, spectra.shape[0]-1)  
ax2.set_title(r'Phase [deg] @ Period = %0.2f minutes' % 4, y = 1.01, fontsize=17)

# was getting error "'AxesImage' object is not iterable"
# - found: "Each element in img needs to be a sequence of artists, not a single artist."
idx = (np.abs(f_fit - 1./(4*60))).argmin()
param = np.copy(spectra[:,:,idx])  # set initial heatmap to power law index     
pflat = np.reshape(param, (param.shape[0]*param.shape[1]))
pNaN = pflat[~np.isnan(pflat)]
h_min = np.percentile(pNaN,1)  # set heatmap vmin to 1% of data (could lower to 0.5% or 0.1%)
h_max = np.percentile(pNaN,99)  # set heatmap vmax to 99% of data (could up to 99.5% or 99.9%)
im2, = ([ax2.imshow(param, cmap=c_map, interpolation='nearest', vmin=h_min, vmax=h_max)])  

global cbar2
# design colorbar for heatmaps
divider2 = make_axes_locatable(ax2)
cax2 = divider2.append_axes("right", size="3%", pad=0.07)
cbar2 = plt.colorbar(im2,cax=cax2)
cbar2.ax.tick_params(labelsize=13, pad=3)   

plt.tight_layout()

axslider = plt.axes([0.15, 0.90, 0.7, 0.04])
#slid_mask = Slider(axslider, 'Frequency', f_fit[0], f_fit[-1], valinit=(1./240))
#slid_mask = Slider(axslider, 'Period', (1./f_fit[-1])/60., (1./f_fit[0])/60., valinit=4., valfmt='%0.2f')
#slid_mask = Slider(axslider, 'Period', (1./f_fit[-1])/60., 50., valinit=4., valfmt='%0.2f')
slid_mask = Slider(axslider, 'Period', (1./f_fit[-1])/60., 20., valinit=4., valfmt='%0.2f')
slid_mask.on_changed(update)



"""
colormapnames = ['segmented map', 'hue-HUSL', 'lum-HSLUV', 'lum-HPLUV']
colormaps = [segmented_cmap, flat_huslmap, hsluv_anglemap, hpluv_anglemap]
for ii, cm in enumerate(colormaps):
    ax = fig.add_subplot(2, 2, ii+1)
    pmesh = ax.pcolormesh(x, y, z/np.pi, 
        cmap = cm, vmin=-1, vmax=1)
    plt.axis([x.min(), x.max(), y.min(), y.max()])
    cbar = fig.colorbar(pmesh)
    cbar.ax.set_ylabel('Phase [pi]')
    ax.set_title( colormapnames[ii] )
plt.show()
"""