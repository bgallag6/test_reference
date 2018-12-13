# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 16:10:04 2018

@author: Brendan
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import f as ff
from matplotlib import cm
import sunpy

plt.rcParams["font.family"] = "Times New Roman"
font_size = 27  # set the font size to be used for all text - titles, tick marks, text, labels  

directory = 'F:'
date = '20130626'
wavelength = 1700
savefig = False

  
# 11-param-list
titles = [r'Power Law Slope-Coefficient [flux] - A', r'(b) Power Law Index n', r'Power Law Tail - C', r'Lorentzian Amplitude [flux] - α', r'(c) Lorentz. Loc. β [min]', r'Lorentzian Width - σ', 'F-Statistic', r'Lorentzian Amplitude Scaled - α', r'$r$-Value: Correlation Coefficient', r'(d) Rollover Period $T_r$ [min]', r'$\chi^2$']
names = ['slope_coeff', 'index', 'tail', 'lorentz_amp', 'lorentz_loc', 'lorentz_wid', 'f_test', 'lorentz_amp_scaled', 'r_value', 'roll_freq', 'chisqr']

# load parameter array and visual images from file tree structure 
h_map = np.load('%s/DATA/Output/%s/%i/param.npy' % (directory, date, wavelength))
visual = np.load('%s/DATA/Output/%s/%i/visual.npy'% (directory, date, wavelength))  

visual = visual[1:-1,1:-1]  # to make same size as heatmaps (if using 3x3 pixel box averaging)  

### for Global EUV Paper
# trim x/y dimensions equally so that resulting region is 1600x1600    
trim_y = int((h_map.shape[1]-1600)/2)
trim_x = int((h_map.shape[2]-1600)/2)
h_map = h_map[:, trim_y:h_map.shape[1]-trim_y, trim_x:h_map.shape[2]-trim_x]  # trim to 1600x1600 (derotate based on mid-file, take off even amounts from both sides)    

x_ticks = [0,200,400,600,800,1000,1200,1400,1600]
y_ticks = [0,200,400,600,800,1000,1200,1400,1600]  
x_ind = [-800,-600,-400,-200,0,200,400,600,800]
y_ind = [800,600,400,200,0,-200,-400,-600,-800] 

fig_width = 12
fig_height = 10

i = 10
    
fig = plt.figure(figsize=(fig_width,fig_height))
ax = plt.gca()  # get current axis -- to set colorbar 
plt.title(r'%s' % (titles[i]), y = 1.02, fontsize=font_size)

h_map[i] *= 3  # variance of 9 spectra
param = h_map[i] 

cmap = cm.get_cmap('jet', 10)  # specify discrete colorscale with 10 intervals 
 
pflat = np.reshape(param, (param.shape[0]*param.shape[1]))
pNaN = pflat[~np.isnan(pflat)]
h_min = np.percentile(pNaN,1)  # set heatmap vmin to 1% of data (could lower to 0.5% or 0.1%)
h_max = np.percentile(pNaN,99)  # set heatmap vmax to 99% of data (could up to 99.5% or 99.9%)

# specify colorbar ticks to be at boundaries of segments
h_range = np.abs(h_max-h_min)
h_step = h_range / 10.
c_ticks = np.zeros((11))
for h in range(11):
    c_ticks[h] = h_min + h_step*h 
    
im = ax.imshow(np.flipud(h_map[i]), cmap = cmap, vmin=h_min, vmax=h_max)
plt.xticks(x_ticks,x_ind,fontsize=font_size)
plt.yticks(y_ticks,y_ind,fontsize=font_size)
ax.tick_params(axis='both', which='major', pad=10)
divider = make_axes_locatable(ax)  # set colorbar to heatmap axis
cax = divider.append_axes("right", size="3%", pad=0.07)
cbar = plt.colorbar(im,cax=cax, format='%0.2f')

cbar.ax.tick_params(labelsize=font_size, pad=5) 
cbar.set_ticks(c_ticks)

if savefig == True:
    plt.savefig('C:/Users/Brendan/Desktop/20130626_%i_chi2.pdf' % wavelength, format='pdf', bbox_inches='tight')
    
    
# plot histogram
flat_param = np.reshape(h_map[i], (h_map[i].shape[0]*h_map[i].shape[1]))

# calculate some statistics
sigma = np.std(flat_param)   

fig = plt.figure(figsize=(fig_width+1,fig_height))
plt.title('%s' % (titles[i]), y = 1.02, fontsize=font_size)  # no date / wavelength
plt.xlabel('%s' % titles[i], fontsize=font_size, labelpad=10)
plt.ylabel('Bin Count', fontsize=font_size, labelpad=10)
plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)
plt.xlim(h_min, h_max)
y, x, _ = plt.hist(flat_param, bins=200, range=(h_min, h_max), edgecolor='k')

n=y[1:-2]
bins=x[1:-2]
elem = np.argmax(n)
bin_max = bins[elem]
plt.ylim(0, y.max()*1.1)         

plt.vlines(bin_max, 0, y.max()*1.1, color='black', linestyle='dotted', linewidth=2., label='mode=%0.4f' % bin_max)  
plt.vlines(0, 0, y.max()*1.1, color='white', linestyle='dashed', linewidth=1.5, label='sigma=%0.4f' % sigma)
legend = plt.legend(loc='upper right', prop={'size':20}, labelspacing=0.35)
for label in legend.get_lines():
    label.set_linewidth(2.0)

if savefig == True:
    plt.savefig('C:/Users/Brendan/Desktop/20130626_%i_chi2_histogram.pdf' % wavelength, format='pdf', bbox_inches='tight')