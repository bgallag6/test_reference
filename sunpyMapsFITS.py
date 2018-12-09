# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 07:56:26 2018

@author: Brendan
"""

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import glob
from sunpy.map import Map
import astropy.units as u
from astropy.coordinates import SkyCoord
from matplotlib.widgets import Button, TextBox
from sunpy.coordinates import frames



def onselect(eclick, erelease):
    'eclick and erelease are the press and release events'
    global x1, y1, x2, y2
    px1, py1 = eclick.xdata, eclick.ydata
    px2, py2 = erelease.xdata, erelease.ydata
  
    sub1 = full_map.pixel_to_world(px1*u.pixel, py1*u.pixel)
    sub2 = full_map.pixel_to_world(px2*u.pixel, py2*u.pixel)
    
    x1 = int(sub1.Tx.value)
    y1 = int(sub1.Ty.value)
    x2 = int(sub2.Tx.value)
    y2 = int(sub2.Ty.value)

    sub_map = full_map.submap(sub1, sub2)
    ax2 = plt.subplot2grid((1,33),(0, 19), colspan=14, rowspan=1, projection=sub_map)
    #ax2 = plt.subplot(122, projection=sub_map)
    sub_map.plot()
    ax2.set_autoscale_on(False)

    t_x1.set_text('Bottom Left: (%ix, %iy)' % (x1, y1))
    t_x2.set_text('Top Right: (%ix, %iy)' % (x2, y2))
    
    plt.draw()
    

"""
############################################
"""

raw_dir = 'S:/FITS/20180211/1700'
processed_dir = 'S:/DATA/20180211/1700'

flist = sorted(glob.glob('%s/*.fits' % raw_dir))

nf = len(flist)

mid_file = nf // 2

sub1 = SkyCoord(-150*u.arcsec, -100*u.arcsec, frame=frames.Helioprojective)
sub2 = SkyCoord(150*u.arcsec, 75*u.arcsec, frame=frames.Helioprojective)


full_map = Map(flist[mid_file]).submap(sub1, sub2)
fm_shape = full_map.data.shape


fig = plt.figure(figsize=(10,6))
plt.subplots_adjust(bottom=0.23)

#ax = plt.subplot(121, projection=full_map)
ax = plt.subplot2grid((1,33),(0, 0), colspan=14, rowspan=1, projection=full_map)
im = full_map.plot() 
ax.set_autoscale_on(False)
ax.set_ylabel('')
ax.set_xlabel('')

#"""
vis = np.load('%s/visual.npy' % processed_dir)
### maybe have full region plotted initially
#ax2 = plt.subplot(122, projection=m1)
ax2 = plt.subplot(122)
ax2.imshow(vis, cmap='sdoaia1700', vmin=100, vmax=3000)
#m1.plot()
#ax2.set_autoscale_on(False)
ax2.set_title('Visual Image')
ax2.set_xlim(0,vis.shape[1])
ax2.set_ylim(0,vis.shape[0])
#"""

global x1, y1, x2, y2

x_y1 = full_map.pixel_to_world(-150*u.pixel, -100*u.pixel)
x_y2 = full_map.pixel_to_world(150*u.pixel, fm_shape[0]*u.pixel)

x1 = int(x_y1.Tx.value)
y1 = int(x_y1.Ty.value)
x2 = int(x_y2.Tx.value)
y2 = int(x_y2.Ty.value)
