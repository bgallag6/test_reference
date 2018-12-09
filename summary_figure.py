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
#date = cfg['date']
#wavelength = 1600
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
  

# for 20160626
y1 = 125
y2 = 530
x1 = 625
x2 = 1030
vis = vis[y1:y2, x1:x2]   
loc_mask = loc_mask[y1:y2, x1:x2]    


#fig, ((ax1,ax2), (ax3,ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(9,6))
#fig, (ax1,ax3) = plt.subplots(nrows=1, ncols=2, figsize=(9,3))
fig, (ax1,ax3,ax5) = plt.subplots(nrows=1, ncols=3, figsize=(16,5))  # for 20130626

plt.suptitle("2013/06/26 — 1700 Å", y=1.02)

v_min = np.percentile(vis,1)
v_max = np.percentile(vis,99) 

ax1.set_title('Averaged Visual Image')
ax1.set_xlim(0, vis.shape[1]-1)
ax1.set_ylim(0, vis.shape[0]-1)
im1 = ax1.imshow(vis, cmap='sdoaia%i' % wavelength, vmin=v_min, vmax=v_max)
ax1.tick_params(axis='both', which='major', pad=3)
divider = make_axes_locatable(ax1)
cax1 = divider.append_axes("right", size="3%", pad=0.07)
cbar = plt.colorbar(im1,cax=cax1)
cbar.ax.tick_params(pad=3)

"""
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

ax2.set_title('Power Law Index')
ax2.set_xlim(0, h_map[1].shape[1]-1)
ax2.set_ylim(0, h_map[1].shape[0]-1)
im2 = ax2.imshow(h_map[1], cmap=c_map, vmin=h_min, vmax=h_max)
ax2.tick_params(axis='both', which='major', pad=10)
divider = make_axes_locatable(ax2)
cax2 = divider.append_axes("right", size="3%", pad=0.07)
cbar = plt.colorbar(im2,cax=cax2, format='%0.2f')
cbar.ax.tick_params(pad=5)
cbar.set_ticks(c_ticks)
"""

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
    
ax3.set_title('Lorentzian Location [min] | f$_{masked}$ = %0.1f%s' % (mask_percent, '%'))
ax3.set_xlim(0, loc_mask.shape[1]-1)
ax3.set_ylim(0, loc_mask.shape[0]-1)
im3 = ax3.imshow(loc_mask, cmap=c_map, vmin=h_min, vmax=h_max)
ax3.tick_params(axis='both', which='major', pad=3)
divider = make_axes_locatable(ax3)
cax3 = divider.append_axes("right", size="3%", pad=0.07)
cbar = plt.colorbar(im3,cax=cax3, format='%0.1f')
cbar.ax.tick_params(pad=3)
cbar.set_ticks(c_ticks)

"""
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

ax4.set_title('Lorentzian FWHM [min] | f$_{masked}$ = %0.1f%s' % (mask_percent, '%'))
ax4.set_xlim(0, fwhm_mask.shape[1]-1)
ax4.set_ylim(0, fwhm_mask.shape[0]-1)
im4 = ax4.imshow(fwhm_mask, cmap=c_map, vmin=h_min, vmax=h_max)
ax4.tick_params(axis='both', which='major', pad=10)
divider = make_axes_locatable(ax4)
cax4 = divider.append_axes("right", size="3%", pad=0.07)
cbar = plt.colorbar(im4,cax=cax4, format='%0.1f')
cbar.ax.tick_params(pad=5)
cbar.set_ticks(c_ticks)
"""

# define combined-fitting function (Model M2)
def LorentzPowerBase(f2, A2, n2, C2, P2, fp2, fw2):
    return A2*f2**-n2 + C2 + P2*(1./ ((np.pi*fw2)*(1.+((np.log(f2)-fp2)/fw2)**2)))

# define Power-Law-fitting function (Model M1)
def PowerLaw(f, A, n, C):
    return A*f**-n + C
        
# define Gaussian-fitting function
def Lorentz(f, P, fp, fw):
    return P*(1./ ((np.pi*fw)*(1.+((np.log(f)-fp)/fw)**2))) 



#plt.savefig('C:/Users/Brendan/Desktop/samplefig20130626.pdf', format='pdf', bbox_inches='tight')
#plt.savefig('C:/Users/Brendan/Desktop/samplefig20130626.png', format='png', bbox_inches='tight')

cube_shape = np.load('%s/derotated_mmap_shape.npy' % directory)
cube = np.memmap('%s/derotated_mmap.npy' % directory, dtype='int16', mode='r', shape=(cube_shape[0], cube_shape[1], cube_shape[2]))
exposure = np.load('%s/exposure.npy' % directory)
time = np.load('%s/time.npy' % directory)
n_segments = 6
timeStep = 24
        
# interpolate timestamps onto default-cadence time-grid
t_interp = np.linspace(0, time[-1], int(time[-1]//timeStep)+1)  
 
# determine frequency values that FFT will evaluate   
n = len(t_interp)
rem = n % n_segments
freq_size = (n - rem) // n_segments

sample_freq = fftpack.fftfreq(freq_size, d=timeStep)
pidxs = np.where(sample_freq > 0)
freqs = sample_freq[pidxs]

spectra_seg = np.zeros((9,len(freqs)))
   
#xx = 350
#yy = 50 
xx = 195
yy = 198
x3 = xx + x1
y3 = yy + y1

count = 0
for m0 in [-1,0,1]:
    for n0 in [-1,0,1]:
        
        #x = x3 + n0 + 1  #offset for 171
        #y = iy + m0 + 6  #offset for 171
        #y = y3 + m0 + 3  #offset for 1700
        x = x3 + n0
        y = y3 + m0
            
        pixmed = cube[:,y,x] / exposure  # extract timeseries + normalize by exposure time   
        
        v_interp = np.interp(t_interp,time,pixmed)  # interpolate pixel-intensity values onto specified time grid
        
        data = v_interp
        
        avg_array = np.zeros((len(freqs)))  # initialize array to hold fourier powers
        
        data = data[0:len(data)-rem]  # trim timeseries to be integer multiple of n_segments
        split = np.split(data, n_segments)  # create split array for each segment
        
        #"""   
        for i in range(n_segments):     
            
          ## perform Fast Fourier Transform on each segment       
          sig = split[i]
          sig_fft = fftpack.fft(sig)
          powers = np.abs(sig_fft)[pidxs]
          norm = len(sig)
          powers = ((powers/norm)**2)*(1./(sig.std()**2))*2   # normalize the power
          avg_array += powers
        
        avg_array /= n_segments  # take the average of the segments            
        
        spectra_seg[count] = avg_array
        count += 1
        
spec_avg = np.average(spectra_seg, axis=0)
spec_std = np.std(spectra_seg, axis=0)

s = spec_avg
f_fit = freqs
        
# assign equal weights to all parts of the curve
df = np.log10(f_fit[1:len(f_fit)]) - np.log10(f_fit[0:len(f_fit)-1])
df2 = np.zeros_like(f_fit)
df2[0:len(df)] = df
df2[len(df2)-1] = df2[len(df2)-2]
ds = df2
#ds = spec_std
 
## fit data to combined power law plus gaussian component model

try:
    # initial guesses for fitting parameters
    M1_low = [-0.002, 0.3, -0.01]
    M1_high = [0.002, 6., 0.01]
    nlfit_l, nlpcov_l = scipy.optimize.curve_fit(PowerLaw, f_fit, s, bounds=(M1_low, M1_high), sigma=ds, method='dogbox')
  
except RuntimeError:
    print("Error M1 - curve_fit failed")
    pass

except ValueError:
    print("Error M1 - inf/NaN")
    pass

A, n, C = nlfit_l  # unpack fitting parameters
 
   
try:                                 

    M2_low = [0., 0.3, -0.01, 0.00001, -6.5, 0.05]
    M2_high = [0.002, 6., 0.01, 0.2, -4.6, 0.8]        
            
    nlfit_gp, nlpcov_gp = scipy.optimize.curve_fit(LorentzPowerBase, f_fit, s, bounds=(M2_low, M2_high), sigma=ds, method='dogbox', max_nfev=3000)

except RuntimeError:
    print("Error M2 - curve_fit failed")
    pass

except ValueError:
    print("Error M2 - inf/NaN")
    pass


A2, n2, C2, P2, fp2, fw2 = nlfit_gp  # unpack fitting parameters
#print nlfit_gp
       
try:           
    nlfit_gp2, nlpcov_gp2 = scipy.optimize.curve_fit(LorentzPowerBase, f_fit, s, p0 = [A2, n2, C2, P2, fp2, fw2], bounds=(M2_low, M2_high), sigma=ds, max_nfev=3000)    
   
except RuntimeError:
    print("Error M2 - curve_fit failed")
    pass

except ValueError:
    print("Error M2 - inf/NaN")
    pass

A22, n22, C22, P22, fp22, fw22 = nlfit_gp2  # unpack fitting parameters     
#print nlfit_gp2
               
# create model functions from fitted parameters    
m1_fit = PowerLaw(f_fit, A, n, C)    
lorentz = Lorentz(f_fit,P22,fp22,fw22)
m2_fit2 = LorentzPowerBase(f_fit, A22,n22,C22,P22,fp22,fw22) 
m1_fit2 = PowerLaw(f_fit, A22,n22,C2)      

ds = spec_std / np.sqrt(3)  # for 1700

residsM1 = (s - m1_fit)
chisqrM1 =  ((residsM1/ds)**2).sum() 

residsM22 = (s - m2_fit2)
chisqrM22 = ((residsM22/ds)**2).sum()
redchisqrM22 = ((residsM22/ds)**2).sum()/float(f_fit.size-6) 
      
#f_test = ((chisqrM1-chisqrM2)/(6-3))/((chisqrM2)/(f.size-6))
f_test2 = ((chisqrM1-chisqrM22)/(6-3))/((chisqrM22)/(f_fit.size-6))

df1, df2 = 3, 6  # degrees of freedom for model M1, M2
p_val = ff.sf(f_test2, df1, df2)

amp_scale2 = PowerLaw(np.exp(fp22), A22, n22, C22)  # to extract the gaussian-amplitude scaling factor

rollover = (1. / ((C22 / A22)**(-1. / n22))) / 60.

fwhm = (1. / (np.exp(fp22+fw22) - np.exp(fp22-fw22))) / 60.


ax5.loglog(f_fit, s, 'k', linewidth=1.5)
ax5.loglog(f_fit, m2_fit2, color='purple', linewidth=1.5, label='Model M2')
ax5.loglog(f_fit, m1_fit2, color='green', linewidth=1.5, label='M2: Power Law')
ax5.loglog(f_fit, lorentz, color='green', linestyle='dashed', linewidth=1.5, label='M2: Lorentzian')
ax5.set_xlabel('Frequency [Hz]')
ax5.set_ylabel('Power')
ax5.set_title('Spectra Fit: Pixel (%ix , %iy)' % (xx, yy))
#ax5.text(0.011, 10**-0.5, r'$n$ = {0:0.2f}'.format(n22))
#ax5.text(0.011, 10**-0.75, r'$\alpha$ = {0:0.2e}'.format(P22))
#ax5.text(0.011, 10**-1.00, r'$\beta$ = {0:0.1f} [min]'.format((1./np.exp(fp22))/60.))
#ax2.text(0.011, 10**-1.5, r'$p$ = {0:0.2e}'.format(p_val), fontsize=font_size-2)
#plt.vlines((0.0093),10**-8,10**1, linestyles='dotted', label='3 minutes')
legend = ax5.legend(loc='lower left')
ax5.set_xlim(10**-4.0, 10**-1.5)
ax5.set_ylim(10**-4., 10**0)   
ax5.plot([np.exp(fp22),np.exp(fp22)], [10**-4,10**0], color='blue', linestyle='dashed')
ax5.text(np.exp(fp22)+1e-3, 10**-1.5, 'Lorentzian Location \n= {0:0.2f} mHz ({1:0.2f} min)'.format(np.exp(fp22)*1000,(1./np.exp(fp22))/60.), color='blue')
ax5.tick_params(axis='both', which='major')
ax1.scatter(xx, yy, s=230, marker='x', c='red', linewidth=3.5)
ax3.scatter(xx, yy, s=230, marker='x', c='red', linewidth=3.5)
ax1.scatter(xx, yy, s=200, marker='x', c='white', linewidth=2.5)
ax3.scatter(xx, yy, s=200, marker='x', c='white', linewidth=2.5)
for label in legend.get_lines():
        label.set_linewidth(2.0)  # the legend line width   

plt.tight_layout()

#plt.savefig('C:/Users/Brendan/Desktop/samplefig20130626C.png', format='png', bbox_inches='tight')