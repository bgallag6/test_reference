# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 21:08:55 2018

@author: Brendan
"""

import glob
import numpy as np
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord
import sunpy
from sunpy.map import Map
from sunpy.coordinates import frames
from sunpy.physics.solar_rotation import calculate_solar_rotate_shift
from sunpy.physics.differential_rotation import diffrot_map
from timeit import default_timer as timer
import time
import datetime
import sys
import os

wavelengths = ['94a', '131a', '171a', '193a', '211a', '304a', '335a', '1600a', '1700a']

dir1 = 'S:/FITS/20120923/171'

flist = sorted(glob.glob('%s/aia*.fits' % dir1))

filename = flist[0]

wave1 = False

for wave in wavelengths:
    if filename.find(wave) != -1:
        wave1 = int(wave[:-1])
if wave1:
    print('wavelength: ',wave1)
else:
    print('No wavelength found.')
        
