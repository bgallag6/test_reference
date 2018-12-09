# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 18:41:21 2018

@author: Brendan
"""

import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
import sunpy
from sunpy.map import Map
from astropy.coordinates import SkyCoord


"""
# check fits image
directory = 'S:'
date = '20130625'
wavelength = 1700

image = '%s/FITS/%s/%i/aia_lev1_1700a_2013_06_25t18_00_31_24z_image_lev1.fits' % (directory, date, wavelength)    

#c3 = SkyCoord((x1-diffrot_longitude)*u.arcsec, y1*u.arcsec, frame=frames.Helioprojective)  
#c4 = SkyCoord((x2+diffrot_longitude)*u.arcsec, y2*u.arcsec, frame=frames.Helioprojective) 


#x1 = x1
#x2 = x2
#y1 = y1
#y2 = y2

m1 = Map('%s' % image)
m1.peek()
"""



"""
#
p1 = np.load('S:/DATA/20180130/1600/dataCube.npy')[18]
cube_shape = np.load('S:/DATA/20180130/1600/check_against/derotated_mmap_shape.npy' )
p2 = np.memmap('S:/DATA/20180130/1600/check_against/derotated_mmap.npy', dtype='int16', mode='r', shape=(cube_shape[0], cube_shape[1], cube_shape[2]))[18]
#p2 = np.load('S:/DATA/20180130/1600/check_against/dl.npy')

p3 = p1-p2

print(np.max(p3), np.min(p3))
"""

# Check / compare files
#dir1 = 'C:/Users/Brendan/Desktop/specFit/images/processed/20120606/1600'
dir1 = 'S:/DATA/20150218/1600'

files = ['visual.npy', 'dataCube.npy', 'specCube.npy', 'timestamps.npy', 'frequencies.npy', 'param.npy']

file = files[1]

check = np.load('%s/%s' % (dir1, file))
#diff = np.load('%s/visual.npy' % dir1)
#old = np.load('%s/compare3x3/visual.npy' % dir1)

#diff = np.load('S:/DATA/20101028/1600/dataCube.npy')
#old = np.load('S:/DATA/20101028/1600/lorentz/dataCube.npy')
#new = np.transpose(old, (2, 0, 1))

#diff = np.load('C:/Users/Brendan/Desktop/specFit/images/processed/20120606/1600/dataCube.npy')
#old = np.load('C:/Users/Brendan/Desktop/specFit/images/processed/20120606/1600/compare/dataCube.npy')

#z = diff-old

#print(z.max(), z.min())   

#"""
#ts1 = diff[0,0]
#ts2 = old[:,100,100]
#ts2 = new[0,0]

#z2 = ts1 - ts2
#z2 = diff-old

#print(z2.max(), z2.min())
#"""

#diff = np.load('C:/Users/Brendan/Desktop/specFit/images/processed/20120606/1600/visual.npy')
#old = np.load('C:/Users/Brendan/Desktop/specFit/images/processed/20120606/1600/compare/visual.npy')

#z = diff-old

#print(z.max(), z.min()) 
