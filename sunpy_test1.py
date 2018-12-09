# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 08:13:27 2018

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



class Index(object):
    
    def saveFig(self, event):
        global x1, y1, x2, y2
        print("Saving coordinates: (%i, %i), (%i, %i)." % (x1, y1, x2, y2), flush=True)
        """
        ### placeholder for saving coordinates to config file
        """

def onselect(eclick, erelease):
    'eclick and erelease are the press and release events'
    global x1, y1, x2, y2
    px1, py1 = eclick.xdata, eclick.ydata
    px2, py2 = erelease.xdata, erelease.ydata
    print(px1,py1)
  
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

def onclick(event):
    global px1, py1
    px1, py1 = event.xdata, event.ydata
    print(px1,py1)
    del ax2.collections[:]
    plt.draw()
    #plt.draw()
    
    ax2.scatter(int(px1),int(py1), s=200, marker='x', c='white', linewidth=2.5)
  
    #sub1 = full_map.pixel_to_world(px1*u.pixel, py1*u.pixel)
    #sub2 = full_map.pixel_to_world(px2*u.pixel, py2*u.pixel)


"""
############################################
"""

raw_dir = 'S:/FITS/20180211/1700'
#raw_dir = './images/raw/20120606/1600/fits'

flist = sorted(glob.glob('%s/*.fits' % raw_dir))

nf = len(flist)

mid_file = nf // 2

sub1 = SkyCoord(-150*u.arcsec, -100*u.arcsec, frame=frames.Helioprojective)
sub2 = SkyCoord(150*u.arcsec, 75*u.arcsec, frame=frames.Helioprojective)

full_map = Map(flist[mid_file]).submap(sub1, sub2)
fm_shape = full_map.data.shape


fig1 = plt.figure(figsize=(10,6))

#ax = plt.subplot(121, projection=full_map)
ax = plt.subplot2grid((1,33),(0, 0), colspan=14, rowspan=1, projection=full_map)
im = full_map.plot(picker=True) 
ax.set_autoscale_on(False)

fig1.canvas.mpl_connect('button_press_event', onclick)

processed_dir = 'S:/DATA/20180211/1700'
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

#xtick2 = ax2.get_xticks()
#ytick2 = ax2.get_yticks()

xind = [0, vis.shape[1]]
yind = [0, vis.shape[0]]

vals = np.array([5, 10, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250])

xticks = [int(sub1.Tx.value), int(sub2.Tx.value)]
yticks = [int(sub1.Ty.value), int(sub2.Ty.value)]

xarcrange = int(sub2.Tx.value) - int(sub1.Tx.value)
yarcrange = int(sub2.Ty.value) - int(sub1.Ty.value)

step = vals[np.abs(vals - (xarcrange/5)).argmin()]

"""
## if 0, then put 0 and go off of that.  
## otherwise, variable to clean numbers: 50, 75, 100, 125, 150
## how to find good start/end points?
## might be better to just get scale of images
"""

plt.xticks(xind, xticks)
plt.yticks(yind, yticks)

ax2.set_xlabel('Helioprojective Longitude (Solar-X) [arcsec]')
ax2.set_ylabel('Helioprojective Latitude (Solar-Y) [arcsec]')

#ax2.set_xlim(xtick10[0],xtick10[-1])
#ax2.set_ylim(ytick10[0],ytick10[-1])

#"""

"""
data = vis
header = {'cdelt1': vis.shape[0], 'cdelt2': vis.shape[1], 'telescop':'sunpy'}
my_map = Map(data, header)

plt.figure()
my_map.plot(projection=my_map)
"""