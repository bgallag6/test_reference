# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 19:41:43 2018

@author: Brendan
"""

"""
convert time to timestamps
exposure to exposures
derotated_mmap to dataCube
spectra_mmap to specCube
3x3_stddev_mmap to specUnc

change DATA folder to Processed
get rid of temp and output folders

"""
import numpy as np
from scipy import fftpack
import os
import shutil

"""
## move date directories in Output folder to DATA folder
source = 'S:/DATA/Output/'
dest1 = 'S:/DATA/'

dateDirs = os.listdir(source)

for f in dateDirs:
    if not os.path.exists(dest1+f):
        try:
            shutil.move(source+f, dest1)
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST: raise
"""

"""
## move contents of date directories in Temp folder to date directory
source = 'S:/DATA/Temp/'
dest1 = 'S:/DATA/'

## 
# 1) date
# 2) wavelength
# 3) contents
# 4) matching content

dateDirs = os.listdir(source)

for f in dateDirs:
    if not os.path.exists(dest1+f):
        try:
            shutil.move(source+f, dest1)
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST: raise
    else:
        dateContents = os.listdir(source+f)
        
        for f2 in dateContents:
            if not os.path.exists(dest1+f+'/'+f2):
                try:
                    shutil.move(source+f+'/'+f2, dest1+f+'/')
                except OSError as exc: # Guard against race condition
                    if exc.errno != errno.EEXIST: raise
            else:
                sub1Contents = os.listdir(source+f+'/'+f2)
                
                for f3 in sub1Contents:
                    if not os.path.exists(dest1+f+'/'+f2+'/'+f3):
                        try:
                            shutil.move(source+f+'/'+f2+'/'+f3, dest1+f+'/'+f2+'/')
                        except OSError as exc: # Guard against race condition
                            if exc.errno != errno.EEXIST: raise
                    else:
                        sub2Contents = os.listdir(source+f+'/'+f2+'/'+f3)
                        
                        for f4 in sub2Contents:
                            if not os.path.exists(dest1+f+'/'+f2+'/'+f3+'/'+f4):
                                try:
                                    shutil.move(source+f+'/'+f2+'/'+f3+'/'+f4, dest1+f+'/'+f2+'/'+f3+'/')
                                except OSError as exc: # Guard against race condition
                                    if exc.errno != errno.EEXIST: raise
"""                
                
"""
# Rename files and save memory-mapped as normal numpy arrays, create frequencies array
#dateDir = 'S:/DATA/20170803/'

for date in ['20151231']:
    dateDir = 'S:/DATA/'+date+'/'

    waveDir = os.listdir(dateDir)
    
    #for wave in waveDir:
    for wave in ['1700']:
        
        dataDir = dateDir+wave+'/'
    
        if os.path.exists(dataDir+'exposure.npy'):
            try:
                os.rename(dataDir+'exposure.npy', dataDir+'exposures.npy')
                print("Renamed %s --> %s" % (dataDir+'exposure.npy', dataDir+'exposures.npy'))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST: raise
                
        if os.path.exists(dataDir+'time.npy'):
            try:
                os.rename(dataDir+'time.npy', dataDir+'timestamps.npy')
                print("Renamed %s --> %s" % (dataDir+'time.npy', dataDir+'timestamps.npy'))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST: raise
        
        if os.path.exists(dataDir+'derotated_mmap.npy'):    
            # load memory-mapped array as read-only
            cube_shape = np.load(dataDir+'derotated_mmap_shape.npy')
            cube = np.memmap(dataDir+'derotated_mmap.npy', dtype='int16', mode='r', shape=tuple(cube_shape))
            
            npCube = np.array(cube)
            np.save(dataDir+'dataCube.npy', npCube)
            
            del cube
            del npCube
            
            os.remove(dataDir+'derotated_mmap_shape.npy')
            os.remove(dataDir+'derotated_mmap.npy')
            print("Saved: %s \nDeleted: %s \nDeleted: %s" % (dataDir+'dataCube.npy', dataDir+'derotated_mmap_shape.npy', dataDir+'derotated_mmap.npy'))
            
        if os.path.exists(dataDir+'3x3_stddev_mmap.npy'):    
            # load memory-mapped array as read-only
            cube_shape = np.load(dataDir+'spectra_mmap_shape.npy')
            cube = np.memmap(dataDir+'3x3_stddev_mmap.npy', dtype='float64', mode='r', shape=tuple(cube_shape))
            
            npCube = np.array(cube)
            np.save(dataDir+'specUnc.npy', npCube)
            
            del cube
            del npCube
            
            os.remove(dataDir+'3x3_stddev_mmap.npy')
            print("Saved: %s \nDeleted: %s" % (dataDir+'specUnc.npy', dataDir+'3x3_stddev_mmap.npy'))
            
        if os.path.exists(dataDir+'spectra_mmap.npy'):    
            # load memory-mapped array as read-only
            cube_shape = np.load(dataDir+'spectra_mmap_shape.npy')
            cube = np.memmap(dataDir+'spectra_mmap.npy', dtype='float64', mode='r', shape=tuple(cube_shape))
            
            npCube = np.array(cube)
            np.save(dataDir+'specCube.npy', npCube)
            
            del cube
            del npCube
            
            os.remove(dataDir+'spectra_mmap_shape.npy')
            os.remove(dataDir+'spectra_mmap.npy')
            print("Saved: %s \nDeleted: %s \nDeleted: %s" % (dataDir+'specCube.npy', dataDir+'spectra_mmap_shape.npy', dataDir+'spectra_mmap.npy'))
        
        if not os.path.exists(dataDir+'frequencies.npy'):
            if os.path.exists(dataDir+'timestamps.npy'):
                
                dt = np.load(dataDir+'timestamps.npy')
                tDiff = list(np.diff(dt))
                timeStep = max(tDiff, key=tDiff.count)
                
                numSeg = int(np.round((timeStep*len(dt)) / 3600) / 2)
                print("Timespan: %0.3f minutes" % (dt[-1]/60))
                print("Number of segments: %i" % numSeg)
        
                # determine frequencies that the FFT will evaluate 
                dtInterp = np.linspace(0, dt[-1], int(dt[-1]/timeStep)+1)
                  
                n = len(dtInterp)
                rem = len(dtInterp) % numSeg
                freq_size = (len(dtInterp) - rem) // numSeg
                
                sample_freq = fftpack.fftfreq(freq_size, d=timeStep)
                pidxs = np.where(sample_freq > 0)
                freqs = sample_freq[pidxs]
                print("Window Length: %i, Frequencies: %i, Low Period: %0.1f, High Period: %0.1f" % (freq_size, len(freqs), 1./freqs[-1], 1./freqs[0]))
                
                np.save(dataDir+'frequencies.npy', freqs)
                print("Saved: %s" % dataDir+'frequencies.npy')

"""

"""
# reshape dataCube arrays
#dateDir = 'S:/DATA/20170803/'
dates = ['20131015','20140130','20140409','20140601',
         '20140724','20141205','20150218','20150530','20150915',
         '20151231','20160319','20160602','20160917','20170111','20170603',
         '20170803','20170830','20171015','20171104','20171210',
         '20171230','20180130','20180219','20180315','20180429',
         '20180518','20180614']

for date in dates:
#for date in ['20110207']:
    dateDir = 'S:/DATA/'+date+'/'

    waveDir = os.listdir(dateDir)
    
    #for wave in waveDir:
    for wave in ['1600','1700']:
        
        dataDir = dateDir+wave+'/'
    
        if os.path.exists(dataDir+'dataCube.npy'):
            try:
                #os.rename(dataDir+'exposure.npy', dataDir+'exposures.npy')
                cube = np.load(dataDir+'dataCube.npy')
                cube_reshape = np.transpose(cube, (1, 2, 0))  # swap 1st and 3rd dimensions
                print(cube.shape, ' --> ', cube_reshape.shape)
                if list(cube[:,15,25]) == list(cube_reshape[15,25]):
                    print('Cubes are identical; deleting old cube.')
                    os.remove(dataDir+'dataCube.npy')
                    print('Saving transformed dataCube in: %s' % dataDir)
                    np.save(dataDir+'dataCube.npy', cube_reshape)
                
                #print("Renamed %s --> %s" % (dataDir+'exposure.npy', dataDir+'exposures.npy'))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST: raise
"""

"""
# save memory map as numpy
directory = 'S:'
date = '20140827'
wavelength = 1600

# load memory-mapped array as read-only
cube_shape = np.load('%s/DATA/Temp/%s/%i/previous/spectra_mmap_shape.npy' % (directory, date, wavelength))
cube = np.memmap('%s/DATA/Temp/%s/%i/previous/spectra_mmap.npy' % (directory, date, wavelength), dtype='float64', mode='r', shape=tuple(cube_shape))

#cube2 = np.array(cube)

#np.save('C:/Users/Brendan/Desktop/mmap_to_npy.npy', cube2)

#cube2 = np.load('C:/Users/Brendan/Desktop/mmap_to_npy.npy')
"""