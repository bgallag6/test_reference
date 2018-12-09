# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 12:24:43 2018

@author: Brendan
"""

"""
## check whether gauss / lorentz oscillation is same, as for using smaller region
"""

import numpy as np
import matplotlib.pyplot as plt
import sunpy.cm

"""
gauss17 = (1./np.exp(np.load('S:/DATA/Output/20110909/1700/gaussian/param.npy')[4]))/60.
lorentz17diffderot = (1./np.exp(np.load('S:/DATA/Output/20110909/1700/diff_derot/param.npy')[4]))/60.


plt.figure(figsize=(8,8))
#plt.imshow(a16_1, cmap='sdoaia1600', vmin=5, vmax=150)
plt.imshow(gauss17, cmap='jet_r', vmin=3.5, vmax=5.5)

plt.figure(figsize=(8,8))
#plt.imshow(b16_1, cmap='sdoaia1600', vmin=5, vmax=150)
plt.imshow(lorentz17diffderot, cmap='jet_r', vmin=3.5, vmax=5.5)

gauss17_flat = np.reshape(gauss17, (gauss17.shape[0]*gauss17.shape[1]))
lorentz17diffderot_flat = np.reshape(lorentz17diffderot, (lorentz17diffderot.shape[0]*lorentz17diffderot.shape[1]))

plt.figure(figsize=(8,8))
plt.hist(gauss17_flat, bins=100, range=(3.5,5.5), edgecolor='k')
plt.ylim(0,60000)

plt.figure(figsize=(8,8))
plt.hist(lorentz17diffderot_flat, bins=100, range=(3.5,5.5), edgecolor='k')
plt.ylim(0,60000)
"""

lorentz_small = (1./np.exp(np.load('S:/DATA/Output/20110909/1700/param.npy')[4]))/60.

plt.figure(figsize=(8,8))
#plt.imshow(b16_1, cmap='sdoaia1600', vmin=5, vmax=150)
plt.imshow(lorentz_small, cmap='jet_r', vmin=3.5, vmax=5.5)

lorentz_small_flat = np.reshape(lorentz_small, (lorentz_small.shape[0]*lorentz_small.shape[1]))

plt.figure(figsize=(8,8))
plt.hist(lorentz_small_flat, bins=50, range=(3.5,5.5), edgecolor='k')
#plt.ylim(0,60000)