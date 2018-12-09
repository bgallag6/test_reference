# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 08:00:34 2018

@author: Brendan
"""

"""
######################
# run with:
# $ mpiexec -n # python diff_derot_mpi.py    (# = number of processors)
######################
"""

import glob
import matplotlib.pyplot as plt
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
import yaml
import time
import datetime


def datacube(flist_chunk):
    # rebin region to desired fraction 
    def rebin(a, *args):
        shape = a.shape
        lenShape = len(shape)
        factor = np.asarray(shape)/np.asarray(args)
        evList = ['a.reshape('] + \
                 ['args[%d],factor[%d],'%(i,i) for i in range(lenShape)] + \
                 [')'] + ['.mean(%d)'%(i+1) for i in range(lenShape)]
        return eval(''.join(evList))
    
    nf1 = len(flist_chunk)
    
    exposure = np.empty((nf1))
    timestamp = np.empty((nf1))
    #visAvg = np.empty((mapShape[0]-(2*diffLatPix), mapShape[1]-(xminI-(-xminF))))
    visAvg = np.empty((mapShape[0], mapShape[1]))
    
    # image data is int16
    #dCube = np.empty((nf1, mapShape[0]-(2*diffLatPix), mapShape[1]-(xminI-(-xminF))), dtype=np.int16)
    dCube = np.empty((nf1, mapShape[0], mapShape[1]), dtype=np.int16)
    
    start = timer()
    T1 = 0
    
    count = 0
    dimCount = 0
    print(mapShape)

    # loop through datacube and extract timeseries, timestamps, exposures
    for filename in flist_chunk:
        smap = Map(filename).submap(c3, c4)
        print(smap.data.shape)
        exposure[count] = (smap.exposure_time).value
        timestamp[count] = Time(smap.date).jd
        dmap = diffrot_map(smap, time=dt0).data
        #dmap = dmap.data
        """
        if dmap.shape != mapShape:
            dimenDiff = np.array(dmap.shape) - np.array(mapShape)
            dmap = dmap[:dmap.shape[0]-dimenDiff[0], :dmap.shape[1]-dimenDiff[1]]
            #dmap = dmap[diffLatPix:-diffLatPix, xminI:-xminF]
            #plt.figure(figsize=(12,10))
            #plt.imshow(dmap, cmap='sdoaia1700', vmin=500, vmax=2000)
            print(dmap.shape)
            visAvg += (dmap / (smap.exposure_time).value)
            dCube[count] = dmap
            dimCount += 1
        else:
            #dmap = dmap[diffLatPix:-diffLatPix, xminI:-xminF]
            #plt.figure(figsize=(12,10))
            #plt.imshow(dmap, cmap='sdoaia1700', vmin=500, vmax=2000)
            visAvg += (dmap / (smap.exposure_time).value)
            dCube[count] = dmap
            print(dmap.shape)
        
        plt.figure(figsize=(12,10))
        plt.imshow(dmap, cmap='sdoaia1700', vmin=500, vmax=2000)
        
        """
        plt.figure(figsize=(12,10))
        plt.imshow(dmap, cmap='sdoaia1700', vmin=500, vmax=2000)
        
        
        
        count += 1        
        
        # estimate time remaining and print to screen
        T = timer()
        T2 = T - T1
        if count == 0:
            T_init = T - start
            T_est = T_init*nf1  
            T_min, T_sec = divmod(T_est, 60)
            T_hr, T_min = divmod(T_min, 60)
            print("On row %i of %i, est. time remaining: %i:%.2i:%.2i" % 
                  (count, nf1, T_hr, T_min, T_sec), flush=True)
        else:
            T_est2 = T2*(nf1-count)
            T_min2, T_sec2 = divmod(T_est2, 60)
            T_hr2, T_min2 = divmod(T_min2, 60)
            print("On row %i of %i, est. time remaining: %i:%.2i:%.2i" % 
                  (count, nf1, T_hr2, T_min2, T_sec2), flush=True)
        T1 = T
    
    dCube_trim = dCube[:, diffLatPix:-diffLatPix, xminI:-xminF]
    visAvg = visAvg[diffLatPix:-diffLatPix, xminI:-xminF]
    print(count,dimCount)
    #np.save('%s/chunk_%i_of_%i' % (datDir, rank+1, size), dCube_trim)
    vis = dmap
    #return dCube_trim
    return exposure, timestamp, visAvg, vis
  

##############################################################################
##############################################################################

 
imDir = 'S:/FITS/20180130/1700'
datDir = 'S:/DATA/20180130/1700'
sub_reg_coords = [-375, 375, -375, 375]

# set variables from command line
x1,x2,y1,y2 = sub_reg_coords

# create a list of all the fits files
flist = sorted(glob.glob('%s/aia*.fits' % imDir))
nf = len(flist)

# Select the middle image, to derotate around
mid_file = np.int(np.floor(nf / 2))

# make mapcube containing first and last maps & calculate derotation shifts
# if not centered at 0deg long, shift amount wont be enough -- 
# maybe only calculate latitude, use other method for longitude trim
mc_shifts = []
mapI = Map(flist[0])
#mapM = Map(flist[mid_file])
mapF = Map(flist[-1])

mc_shifts.append(mapI)
#mc_shifts.append(mapM)
mc_shifts.append(mapF)
new_mapcube1 = Map(mc_shifts, cube=True)
shifts = calculate_solar_rotate_shift(new_mapcube1, layer_index=0)
print(shifts)

# compute longitude / latitude shift over timespan
diffLon = np.abs(np.floor((np.floor(shifts['x'][1].value)/2.)))

# calculate rotation amount in pixels
diffLonPix = diffLon / (mapI.scale)[0].value

# calculate total latitude shift in pixels, x2 since underestimated?
diffLatPix = int((np.abs((shifts['y'][1].value)) / (mapI.scale)[1].value) * 2.)
#print(diffLon, diffLonPix)
#print(diffLatPix*mapI.scale[1].value, diffLatPix)

if diffLatPix == 0:
    diffLatPix = 5  # value of zero messes with next steps
    

c3 = SkyCoord((x1-diffLon)*u.arcsec, y1*u.arcsec, frame=frames.Helioprojective)  
c4 = SkyCoord((x2+diffLon)*u.arcsec, y2*u.arcsec, frame=frames.Helioprojective) 

# get middle frame subregion & time to anchor derotation
midmap = Map(flist[mid_file]).submap(c3,c4)
dt0 = midmap.date
mapShape = midmap.data.shape

# calculate pixels to trim based off of what actual derotation trims 
# *for some reason this result is different than method above
diffMapI = diffrot_map(Map(flist[0]).submap(c3, c4), time=dt0).data
diffMapF = diffrot_map(Map(flist[-1]).submap(c3, c4), time=dt0).data

"""
diffMapI2 = np.fliplr(diffMapI.data)[diffLatPix:-diffLatPix]
diffMapF2 = (diffMapF.data)[diffLatPix:-diffLatPix]
"""

"""
plt.figure(figsize=(12,10))
plt.imshow(diffMapI, cmap='sdoaia1600', vmin=50, vmax=200)

plt.figure(figsize=(12,10))
plt.imshow(diffMapF, cmap='sdoaia1600', vmin=50, vmax=200)
"""

xminindI = np.argmin(np.fliplr(diffMapI), axis=1)[diffLatPix:-diffLatPix]
xminindF = np.argmin(diffMapF, axis=1)[diffLatPix:-diffLatPix]

xminI = mapShape[1] - np.min(xminindI)  
xminF = mapShape[1] - np.min(xminindF)


## split data and send to processors 
chunks = np.array_split(flist, 4)

subcube = chunks[2][8:12]

start = timer()

ex, t, v_avg, vis = datacube( subcube )

"""
tArr -= tArr[0]  # calculate time since first image
tArr = np.around(tArr*86400)  # get timestamps in seconds
  
# Calculate averaged visual image
for j in range(len(all_v_avg)):
    if j == 0:
        vAvgArr = all_v_avg[j]
    else:
        vAvgArr += all_v_avg[j]
vAvgArr /= nf
"""