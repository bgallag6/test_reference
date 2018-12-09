# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 08:45:37 2018

@author: Brendan
"""

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import SkyCoord
from sunpy.map import Map
from sunpy.coordinates import frames


    
imDir = 'S:/FITS/20180518/1700'
datDir = 'S:/DATA/20180518/1700'

sub_reg_coords = [-375, 375, -375, 375]
x1,x2,y1,y2 = sub_reg_coords

diffLon = 20.  

c3 = SkyCoord((x1-diffLon)*u.arcsec, y1*u.arcsec, frame=frames.Helioprojective)  
c4 = SkyCoord((x2+diffLon)*u.arcsec, y2*u.arcsec, frame=frames.Helioprojective) 

dCube = np.empty((5, 1225, 1289), dtype=np.int16)

f1 = 'S:/FITS/20180518/1700/aia_lev1_1700a_2018_05_18t02_41_17_24z_image_lev1.fits'
f2 = 'S:/FITS/20180518/1700/aia_lev1_1700a_2018_05_18t02_41_41_24z_image_lev1.fits'
f3 = 'S:/FITS/20180518/1700/aia_lev1_1700a_2018_05_18t02_42_05_24z_image_lev1.fits'

map1 = Map(f1)
print(map1.data.shape)
smap1 = map1.submap(c3, c4)
print(smap1.data.shape)
plt.figure(figsize=(12,10))
plt.imshow(smap1.data, cmap='sdoaia1700', vmin=500, vmax=2000)

map2 = Map(f2)
print(map2.data.shape)
smap2 = map2.submap(c3, c4)
print(smap2.data.shape)
plt.figure(figsize=(12,10))
plt.imshow(smap2.data, cmap='sdoaia1700', vmin=500, vmax=2000)

map3 = Map(f3)
print(map3.data.shape)
smap3 = map3.submap(c3, c4)
print(smap3.data.shape)
smap3 = smap3[:100,:100]
plt.figure(figsize=(12,10))
plt.imshow(smap3.data, cmap='sdoaia1700', vmin=500, vmax=2000)

dCube[0] = smap1.data
dCube[1] = smap2.data
dCube[2] = smap3.data