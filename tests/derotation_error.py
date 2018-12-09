# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 12:15:31 2018

@author: Brendan
"""

import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import SkyCoord
from sunpy.map import Map
from sunpy.coordinates import frames
from sunpy.physics.solar_rotation import calculate_solar_rotate_shift
from sunpy.physics.differential_rotation import diffrot_map




sub_reg_coords = [-375, 375, -375, 375]
x1,x2,y1,y2 = sub_reg_coords

c3 = SkyCoord((x1-20)*u.arcsec, y1*u.arcsec, frame=frames.Helioprojective)  
c4 = SkyCoord((x2+20)*u.arcsec, y2*u.arcsec, frame=frames.Helioprojective) 



f1 = 'S:/FITS/20180429/1600/aia_lev1_1600a_2018_04_29t00_00_16_24z_image_lev1.fits'
f2 = 'S:/FITS/20180429/1600/aia_lev1_1600a_2018_04_29t01_59_52_24z_image_lev1.fits'
f3 = 'S:/FITS/20180429/1600/aia_lev1_1600a_2018_04_29t03_59_28_24z_image_lev1.fits'

# get middle frame subregion & time to anchor derotation
midmap = Map(f2).submap(c3,c4)
dt0 = midmap.date
mapShape = midmap.data.shape

# calculate pixels to trim based off of what actual derotation trims 
mapI = Map(f1).submap(c3, c4)
mapF = Map(f3).submap(c3, c4)
print(mapI.date)
print(mapF.date)

# calculate derotation shifts
mc_shifts = []
mc_shifts.append(mapI)
mc_shifts.append(mapF)
new_mapcube1 = Map(mc_shifts, cube=True)
shifts = calculate_solar_rotate_shift(new_mapcube1, layer_index=0)
print(shifts['x'][1].value)

diffMapI = diffrot_map(mapI, time=dt0).data
diffMapF = diffrot_map(mapF, time=dt0).data

fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12,6))
ax1.set_title('First Frame')
ax1.imshow(diffMapI, cmap='sdoaia1600', vmin=50, vmax=200)
ax2.set_title('Last Frame')
ax2.imshow(diffMapF, cmap='sdoaia1600', vmin=50, vmax=200)





#f1 = 'S:/FITS/20180407/1600/aia_lev1_1600a_2018_04_07t00_00_16_24z_image_lev1.fits'
f1 = 'S:/FITS/20180407/1600/aia_lev1_1600a_2018_04_07t00_00_40_24z_image_lev1.fits'
#f2 = 'S:/FITS/20180407/1600/aia_lev1_1600a_2018_04_07t01_59_52_24z_image_lev1.fits'
f2 = 'S:/FITS/20180407/1600/aia_lev1_1600a_2018_04_07t02_00_40_24z_image_lev1.fits'
#f3 = 'S:/FITS/20180407/1600/aia_lev1_1600a_2018_04_07t03_59_28_24z_image_lev1.fits'
f3 = 'S:/FITS/20180407/1600/aia_lev1_1600a_2018_04_07t03_59_04_24z_image_lev1.fits'  

# get middle frame subregion & time to anchor derotation
midmap = Map(f2).submap(c3,c4)
dt0 = midmap.date
mapShape = midmap.data.shape

# calculate pixels to trim based off of what actual derotation trims 
# *for some reason this result is different than method above
mapI = Map(f1).submap(c3, c4)
mapF = Map(f3).submap(c3, c4)
print(mapI.date)
print(mapF.date)

mc_shifts = []
mc_shifts.append(mapI)
mc_shifts.append(mapF)
new_mapcube1 = Map(mc_shifts, cube=True)
shifts = calculate_solar_rotate_shift(new_mapcube1, layer_index=0)
print(shifts['x'][1].value)

diffMapI = diffrot_map(mapI, time=dt0).data
diffMapF = diffrot_map(mapF, time=dt0).data

fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12,6))
ax1.set_title('First Frame')
ax1.imshow(diffMapI, cmap='sdoaia1600', vmin=50, vmax=200)
ax2.set_title('Last Frame')
ax2.imshow(diffMapF, cmap='sdoaia1600', vmin=50, vmax=200)