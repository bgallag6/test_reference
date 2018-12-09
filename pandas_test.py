# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 21:06:57 2018

@author: Brendan
"""

from pandas import DataFrame, read_csv
import pandas as pd
import glob
from sunpy.map import Map
import astropy.units as u
import numpy as np
from timeit import default_timer as timer


dbLoc = 'C:/Users/Brendan/Desktop/fits_files.csv'
df = pd.read_csv(dbLoc)

#"""
raw_dir = 'S:/FITS/20110601/1700'

flist = sorted(glob.glob('%s/*.fits' % raw_dir))

def getProperties(filename):
    fmap = Map(filename)
    mapDate = fmap.date
    mapExposure = fmap.exposure_time.value
    mapInst = fmap.instrument
    mapXScale = fmap.scale[0].value
    mapYScale = fmap.scale[1].value
    mapWave = fmap.wavelength.value
    return filename, mapDate, mapWave, mapExposure, mapXScale, mapYScale, mapInst

dataList = []

start = timer()

for file in flist:
    mapProp = getProperties(file)
    dataList.append(mapProp)


df = pd.DataFrame(data = dataList, columns=['Filename', 'Date', 'Wavelength', 'Exposure', 'X-scale', 'Y-scale', 'Instrument'])

#df.to_csv('fits_files.csv', index=False)
df.to_csv('fits_files.csv', mode='a', index=False, header=False)

print(timer() - start)
#"""



