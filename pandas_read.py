# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 20:55:20 2018

@author: Brendan
"""

from pandas import DataFrame, read_csv
import pandas as pd
import glob
from sunpy.map import Map
import astropy.units as u
import numpy as np
from timeit import default_timer as timer
import matplotlib.pyplot as plt


#dbLoc = 'C:/Users/Brendan/Desktop/fits_files.csv'
#df = pd.read_csv(dbLoc)

#sets = df[0::1200]['Date']

sample = df[0::4800]

paths = sample['Filename']
xscale = sample['X-scale']

for file in paths:
    fmap = Map(file)
    fmap.peek()
    
for scale in xscale:
    print(scale)