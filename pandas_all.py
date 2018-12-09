# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 16:08:07 2018

@author: Brendan
"""

import pandas as pd
import glob
from sunpy.map import Map
from timeit import default_timer as timer

def getProperties(filename):
    fmap = Map(filename)
    mapDate = fmap.date
    mapExposure = fmap.exposure_time.value
    mapInst = fmap.instrument
    mapXScale = fmap.scale[0].value
    mapYScale = fmap.scale[1].value
    mapWave = fmap.wavelength.value
    return filename, mapDate, mapWave, mapExposure, mapXScale, mapYScale, mapInst

def addDataset(raw_dir): 
    flist = sorted(glob.glob('%s/*.fits' % raw_dir))
    
    dataList = []
    
    start = timer()

    for file in flist:
        mapProp = getProperties(file)
        dataList.append(mapProp)
    
    df = pd.DataFrame(data = dataList, columns=['Filename', 'Date', 'Wavelength', 'Exposure', 'X-scale', 'Y-scale', 'Instrument'])
    
    df.to_csv('fits_files.csv', mode='a', index=False, header=False)
    
    print(timer() - start)

#"""    
dates = ['20111030','20111110','20120111','20120319','20120621',
         '20120914','20121227','20130301','20130618','20131015','20140130',
         '20140409','20140601','20140708','20140724','20141024','20141205',
         '20150218','20150530','20150915','20151231','20160319','20160602',
         '20160917','20170111','20170603','20170803','20170830','20170906',
         '20171015','20171104','20171210','20171230','20180130','20180219',
         '20180304','20180315','20180429','20180518','20180614'] 
#"""
#dates = ['20110909']
wavelengths = [1600,1700]

for date in dates:
    for wavelength in wavelengths:
        directory = 'S:/FITS/%s/%i' % (date, wavelength)
        addDataset(directory)