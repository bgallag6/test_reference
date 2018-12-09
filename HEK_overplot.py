# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 15:41:29 2018

@author: Brendan
"""

from __future__ import print_function, division

from datetime import timedelta
import numpy as np

import matplotlib.pyplot as plt

import astropy.units as u
from astropy.coordinates import SkyCoord

import sunpy.map

from sunpy.net import hek
from sunpy.time import parse_time
from sunpy.coordinates import frames
from sunpy.physics.differential_rotation import solar_rotate_coordinate


aia_map = sunpy.map.Map('C:/Users/Brendan/Desktop/20120923/aia_lev1_171a_2013_06_26t00_00_11_34z_image_lev1.fits')

hek_client = hek.HEKClient()

start_time = aia_map.date - timedelta(hours=2)
end_time = aia_map.date + timedelta(hours=2)
responses = hek_client.search(hek.attrs.Time(start_time, end_time),
                              hek.attrs.CH, hek.attrs.FRM.Name == 'SPoCA')

"""
area = 0.0
for i, response in enumerate(responses):
    if response['area_atdiskcenter'] > area and np.abs(response['hgc_y']) < 40.0:
        area = response['area_atdiskcenter']
        response_index = i
"""

response_index = 0
        
ch = responses[response_index]
p1 = ch["hpc_boundcc"][9:-2]
p2 = p1.split(',')
p3 = [v.split(" ") for v in p2]
ch_date = parse_time(ch['event_starttime'])


ch_boundary = SkyCoord(
    [(float(v[0]), float(v[1])) * u.arcsec for v in p3],
    obstime=ch_date,
    frame=frames.Helioprojective)
rotated_ch_boundary = solar_rotate_coordinate(ch_boundary, aia_map.date)



fig = plt.figure()
ax = plt.subplot(projection=aia_map)
aia_map.plot(axes=ax)
ax.plot_coord(rotated_ch_boundary, color='c')
ax.set_title('{:s}\n{:s}'.format(aia_map.name, ch['frm_specificid']))
plt.colorbar()
plt.show()

#responses[0]['event_coord1']