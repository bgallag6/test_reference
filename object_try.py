# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 20:33:04 2018

@author: Brendan
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit as Fit
import sunpy
import sunpy.cm
from scipy import fftpack
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.widgets import Button, Slider
from scipy.stats import f as ff
from scipy.stats.stats import pearsonr
import os
import yaml
import glob
from sunpy.map import Map



class cPlot(object):
    
    ax1title = ''
    ax2title = ''
    ax3title = ''
    
    def __init__(self):
        # create figure with heatmap and spectra side-by-side subplots
        self.fig1 = plt.figure(figsize=(18,9))
        
    def ax1_title(self, text):
        self.ax1_title0.set_text(text)
    
    """    
    def ax1setup(self):
        self.ax1 = plt.gca()
        self.ax1 = plt.subplot2grid((30,31),(4, 1), colspan=14, rowspan=16)
        
        self.im = self.ax1.plot([1,2,3,4],[4,3,2,1])
        
        self.ax1_text = ''
    
        self.ax1_title0 = self.ax1.set_title(self.ax1_text, y = 1.01, fontsize=17)
    """    
    class ax1():
        def __init__(self):
            self = plt.gca()
            self = plt.subplot2grid((30,31),(4, 1), colspan=14, rowspan=16)
            self.im = self.plot([1,2,3,4],[4,3,2,1])
            self.ax1_text = 'this'
            self.ax1_title0 = self.set_title(self.ax1_text, y = 1.01, fontsize=17)
             
        #def remove1(self):
        #    self.remove()
            
        # destructor
    	 
            
    class ax2():
        def __init__(self):
            self = plt.gca()
            self = plt.subplot2grid((30,31),(4, 17), colspan=14, rowspan=16)
            self.im = self.plot([1,2,3,4],[4,3,2,1])
            self.ax2_text = ''
            self.ax2_title0 = self.set_title(self.ax2_text, y = 1.01, fontsize=17)
            
    def ax1remove(self):
        cPlot.ax1.cla()
        
cp = cPlot()
#cp.ax1setup()
#cp.ax1_title('this?')