# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 14:12:34 2018

@author: Brendan
"""

import numpy as np
import scipy.signal
from scipy import signal
import scipy.misc
from timeit import default_timer as timer
from mpi4py import MPI
from scipy import fftpack
import yaml
import time
import datetime
import sys
import os

with open('specFit_config_test.yaml', 'r') as stream:
    cfg4 = yaml.load(stream)

directory = cfg['processed_dir']
date = cfg['date']
wavelength = cfg['wavelength']
mmap_datacube = cfg['mmap_datacube']
n_segments = cfg['num_segments']  # break data into # segments of equal length
tStep = cfg["time_step"]
