# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 16:25:49 2018

@author: Brendan
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import f as ff
from matplotlib import cm
import yaml
import sunpy.cm
import sys
import os 
import scipy.signal
from scipy import fftpack

plt.rcParams["font.family"] = "Times New Roman"
font_size = 27

#with open('specFit_config.yaml', 'r') as stream:
#    cfg = yaml.load(stream)

#directory = cfg['processed_dir']
#date = cfg['date']
#wavelength = cfg['wavelength']
#savefig = cfg['save_fig']

#directory = 'C:/Users/Brendan/Desktop/specFit/test/validation/Processed/20120606/1600'
directory = 'S:/DATA/20130626/1700'
wavelength = 1700
savefig = False

# 11-param-list
titles = [r'Power Law Slope-Coefficient [flux] - A', r'Power Law Index n', 
          r'Power Law Tail - C', r'Lorentzian Amplitude [flux] - α', 
          r'Lorentz. Loc. β [min]', r'Lorentzian Width - σ', 'F-Statistic', 
          r'Lorentzian Amplitude Scaled - α', r'$r$-Value: Correlation Coefficient', 
          r'Rollover Period $T_r$ [min]', r'$\chi^2$']
names = ['slope_coeff', 'index', 'tail', 'lorentz_amp', 'lorentz_loc', 
         'lorentz_wid', 'f_test', 'lorentz_amp_scaled', 'r_value', 'roll_freq', 'chisqr']

# load parameter/heatmap array 
h_map = np.load('%s/param.npy' % directory)  

# generate p-value heatmap + masked Lorentzian component heatmaps
dof1, dof2 = 3, 6  # degrees of freedom for model M1, M2
fstat = np.copy(h_map[6])
fstat[np.where(np.isnan(fstat))] = 1.
p_val = ff.sf(fstat, dof1, dof2)

mask_thresh = 0.005  # significance threshold
   
p_mask = np.copy(p_val)
amp_mask = np.copy(h_map[3])
loc_mask = np.copy(h_map[4])
wid_mask = np.copy(h_map[5])    

# mask the Lorenztian component arrays with NaNs if above threshold 
p_mask[p_val > mask_thresh] = np.NaN
amp_mask[p_val > mask_thresh] = np.NaN
loc_mask[p_val > mask_thresh] = np.NaN
wid_mask[p_val > mask_thresh] = np.NaN    

# determine percentage of region masked 
count = np.count_nonzero(np.isnan(p_mask))   
total_pix = p_val.shape[0]*p_val.shape[1]
mask_percent = ((np.float(count))/total_pix)*100

fwhm_mask = (1. / (np.exp(loc_mask+wid_mask) - np.exp(loc_mask-wid_mask))) / 60. 

# convert Lorentzian location to minutes
h_map[4] = (1./np.exp(h_map[4]))/60.               
loc_mask = (1./np.exp(loc_mask))/60. 

#plots = [h_map[1], loc_mask, fwhm_mask]
plots = [h_map[1], loc_mask, amp_mask]

# generate visual images
vis = np.load('%s/visual.npy' % directory)  
vis = vis[1:-1,1:-1]  # make same size as heatmaps (if using 3x3 averaging)
  

# for 20130626
y1 = 125
y2 = 530
x1 = 625
x2 = 1030
vis = vis[y1:y2, x1:x2]   
loc_mask = loc_mask[y1:y2, x1:x2]    


fig, ((ax1,ax2), (ax3,ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(7,6))
#fig, (ax1,ax3) = plt.subplots(nrows=1, ncols=2, figsize=(9,3))
#fig, (ax1,ax3,ax5) = plt.subplots(nrows=1, ncols=3, figsize=(16,5))  # for 20130626

plt.suptitle("2013/06/26 — 1700 Å", y=1.02)

v_min = np.percentile(vis,1)
v_max = np.percentile(vis,99) 

ax1.set_title('Averaged Visual Image', y=1.01)
ax1.set_xlim(0, vis.shape[1]-1)
ax1.set_ylim(0, vis.shape[0]-1)
im1 = ax1.imshow(vis, cmap='sdoaia%i' % wavelength, vmin=v_min, vmax=v_max)
ax1.tick_params(axis='both', which='major', pad=3)
divider = make_axes_locatable(ax1)
cax1 = divider.append_axes("right", size="3%", pad=0.07)
cbar = plt.colorbar(im1,cax=cax1)
cbar.ax.tick_params(pad=3)


pflat = np.reshape(h_map[1], (h_map[1].shape[0]*h_map[1].shape[1]))
pNaN = pflat[~np.isnan(pflat)]
h_min = np.percentile(pNaN,1)
h_max = np.percentile(pNaN,99)

# specify colorbar ticks to be at boundaries of segments
h_range = np.abs(h_max-h_min)
h_step = h_range / 10.
c_ticks = np.zeros((11))
for h in range(11):
    c_ticks[h] = h_min + h_step*h 
    
c_map = cm.get_cmap('jet', 10)

ax2.set_title('Power Law Index', y=1.01)
ax2.set_xlim(0, h_map[1].shape[1]-1)
ax2.set_ylim(0, h_map[1].shape[0]-1)
im2 = ax2.imshow(h_map[1], cmap=c_map, vmin=h_min, vmax=h_max)
ax2.tick_params(axis='both', which='major', pad=3)
divider = make_axes_locatable(ax2)
cax2 = divider.append_axes("right", size="3%", pad=0.07)
cbar = plt.colorbar(im2,cax=cax2, format='%0.2f')
cbar.ax.tick_params(pad=5)
cbar.set_ticks(c_ticks)


pflat = np.reshape(loc_mask, (loc_mask.shape[0]*loc_mask.shape[1]))
pNaN = pflat[~np.isnan(pflat)]
h_min = np.percentile(pNaN,1)
h_max = np.percentile(pNaN,99)

# specify colorbar ticks to be at boundaries of segments
h_range = np.abs(h_max-h_min)
h_step = h_range / 10.
c_ticks = np.zeros((11))
for h in range(11):
    c_ticks[h] = h_min + h_step*h 
    
c_map = cm.get_cmap('jet_r', 10)
    
ax3.set_title('Lorentz. Location [min] | f$_{masked}$ = %0.1f%s' % (mask_percent, '%'), y=1.01)
ax3.set_xlim(0, loc_mask.shape[1]-1)
ax3.set_ylim(0, loc_mask.shape[0]-1)
im3 = ax3.imshow(loc_mask, cmap=c_map, vmin=h_min, vmax=h_max)
ax3.tick_params(axis='both', which='major', pad=3)
divider = make_axes_locatable(ax3)
cax3 = divider.append_axes("right", size="3%", pad=0.07)
cbar = plt.colorbar(im3,cax=cax3, format='%0.1f')
cbar.ax.tick_params(pad=3)
cbar.set_ticks(c_ticks)


pflat = np.reshape(fwhm_mask, (fwhm_mask.shape[0]*fwhm_mask.shape[1]))
pNaN = pflat[~np.isnan(pflat)]
h_min = np.percentile(pNaN,1)
h_max = np.percentile(pNaN,99)

# specify colorbar ticks to be at boundaries of segments
h_range = np.abs(h_max-h_min)
h_step = h_range / 10.
c_ticks = np.zeros((11))
for h in range(11):
    c_ticks[h] = h_min + h_step*h 
    
c_map = cm.get_cmap('jet_r', 10)

ax4.set_title('Lorentz. FWHM [min] | f$_{masked}$ = %0.1f%s' % (mask_percent, '%'), y=1.01)
ax4.set_xlim(0, fwhm_mask.shape[1]-1)
ax4.set_ylim(0, fwhm_mask.shape[0]-1)
im4 = ax4.imshow(fwhm_mask, cmap=c_map, vmin=h_min, vmax=h_max)
ax4.tick_params(axis='both', which='major', pad=3)
divider = make_axes_locatable(ax4)
cax4 = divider.append_axes("right", size="3%", pad=0.07)
cbar = plt.colorbar(im4,cax=cax4, format='%0.1f')
cbar.ax.tick_params(pad=5)
cbar.set_ticks(c_ticks)


plt.tight_layout()
plt.savefig('C:/Users/Brendan/Desktop/samplefig20130626C2.png', format='png', bbox_inches='tight')