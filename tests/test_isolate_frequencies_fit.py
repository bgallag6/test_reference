# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 09:51:13 2018

@author: Brendan
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RectangleSelector
import matplotlib.patches as patches
import scipy.signal
#matplotlib.use('TkAgg') 	# NOTE: This is a MAC/OSX thing. Probably REMOVE for linux/Win
from matplotlib.widgets import Cursor
from pylab import axvline
import sunpy
from scipy import signal
from scipy import fftpack
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.widgets import Button
from matplotlib.widgets import RadioButtons
from scipy.stats import f as ff
from scipy.stats.stats import pearsonr
import os
from sunpy.map import Map

    
def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]

def plotMap(p):
    param = h_map[p]
    pflat = np.reshape(param, (param.shape[0]*param.shape[1]))
    pNaN = pflat[~np.isnan(pflat)]
    h_min = np.percentile(pNaN,1)  # set heatmap vmin to 1% of data (could lower to 0.5% or 0.1%)
    h_max = np.percentile(pNaN,99)  # set heatmap vmax to 99% of data (could up to 99.5% or 99.9%)
    #h_min = np.percentile(param,1)  # set heatmap vmin to 1% of data (could lower to 0.5% or 0.1%)
    #h_max = np.percentile(param,99)  # set heatmap vmax to 99% of data (could up to 99.5% or 99.9%)
    im = ax1.imshow(param, cmap='jet', interpolation='nearest', vmin=h_min, vmax=h_max, picker=True)
    ax1.set_title(r'%s: %i $\AA$ | %s' % (date_title, wavelength, titles[p]), y = 1.01, fontsize=17)
    plt.colorbar(im,cax=cax)
    plt.draw()

def plotMask(p):
    ax1.clear()
    ax1.set_xlim(0, h_map.shape[2]-1)
    ax1.set_ylim(0, h_map.shape[1]-1)  
    ax1.set_title(r'%s: %i $\AA$ | %s' % (date_title, wavelength, titles[p]), y = 1.01, fontsize=17)
    
    param = h_map[p]
    pflat = np.reshape(param, (param.shape[0]*param.shape[1]))
    pNaN = pflat[~np.isnan(pflat)]
    h_min = np.percentile(pNaN,1)  # set heatmap vmin to 1% of data (could lower to 0.5% or 0.1%)
    h_max = np.percentile(pNaN,99)  # set heatmap vmax to 99% of data (could up to 99.5% or 99.9%)
    #h_min = np.percentile(param,1)  # set heatmap vmin to 1% of data (could lower to 0.5% or 0.1%)
    #h_max = np.percentile(param,99)  # set heatmap vmax to 99% of data (could up to 99.5% or 99.9%)
    
    # generate p-value heatmap + masked Gaussian component heatmaps
    df1, df2 = 3, 6  # degrees of freedom for model M1, M2
    p_val = ff.sf(h_map[6], df1, df2)
    mask_thresh = 0.005  # significance threshold - masked above this value
    param_mask = np.copy(param) 
    param_mask[p_val > mask_thresh] = np.NaN  # mask the Gaussian component arrays with NaNs if above threshold
    im = ax1.imshow(param_mask, cmap='jet', interpolation='nearest', vmin=h_min, vmax=h_max, picker=True)
    plt.colorbar(im,cax=cax)
    plt.draw()

    
class Index(object):
    ind = 0

    def index(self, event):
        global marker
        marker = 1
        plotMap(marker)
        return marker     
        
    def roll(self, event):  # meh, should probably fix this
        global marker
        marker = 2
        paramA = h_map[0]
        paramn = h_map[1]
        paramC = h_map[2]
        param = (paramC/paramA)**(1./paramn)
        param = np.nan_to_num(param)/60.
        h_min = np.percentile(param,1)  # set heatmap vmin to 1% of data (could lower to 0.5% or 0.1%)
        h_max = np.percentile(param,99)  # set heatmap vmax to 99% of data (could up to 99.5% or 99.9%)
        im = ax1.imshow(param, cmap='jet', interpolation='nearest', vmin=h_min, vmax=h_max,  picker=True)
        ax1.set_title(r'%s: %i $\AA$ | %s' % (date_title, wavelength, titles[2]), y = 1.01, fontsize=17)
        plt.colorbar(im,cax=cax)
        plt.draw()      
        
    def lorentz_amp(self, event):
        global marker
        marker = 3
        im = ax1.imshow(p_white, cmap='PiYG', vmin=-1, vmax=1)
        param = h_map[3]
        pflat = np.reshape(param, (param.shape[0]*param.shape[1]))
        pNaN = pflat[~np.isnan(pflat)]
        h_min = np.percentile(pNaN,1)  # set heatmap vmin to 1% of data (could lower to 0.5% or 0.1%)
        h_max = np.percentile(pNaN,99)  # set heatmap vmax to 99% of data (could up to 99.5% or 99.9%)
        #h_min = np.percentile(param,1)  # set heatmap vmin to 1% of data (could lower to 0.5% or 0.1%)
        #h_max = np.percentile(param,99)  # set heatmap vmax to 99% of data (could up to 99.5% or 99.9%)
        im = ax1.imshow(param, cmap='jet', interpolation='nearest', vmin=h_min, vmax=h_max,  picker=True)
        ax1.set_title(r'%s: %i $\AA$ | %s' % (date_title, wavelength, titles[3]), y = 1.01, fontsize=17)
        plt.colorbar(im,cax=cax)
        plt.draw()
    
    def lorentz_loc(self, event):
        global marker
        marker = 4
        im = ax1.imshow(p_white, cmap='PiYG', vmin=-1, vmax=1)
        param = (1./(np.exp(h_map[4]))/60.)
        pflat = np.reshape(param, (param.shape[0]*param.shape[1]))
        pNaN = pflat[~np.isnan(pflat)]
        h_min = np.percentile(pNaN,1)  # set heatmap vmin to 1% of data (could lower to 0.5% or 0.1%)
        h_max = np.percentile(pNaN,99)  # set heatmap vmax to 99% of data (could up to 99.5% or 99.9%)
        #h_min = np.percentile(param,1)  # set heatmap vmin to 1% of data (could lower to 0.5% or 0.1%)
        #h_max = np.percentile(param,99)  # set heatmap vmax to 99% of data (could up to 99.5% or 99.9%)
        #h_min = 1.
        #h_max = 11.
        im = ax1.imshow(param, cmap='jet_r', interpolation='nearest', vmin=h_min, vmax=h_max,  picker=True)
        ax1.set_title(r'%s: %i $\AA$ | %s' % (date_title, wavelength, titles[4]), y = 1.01, fontsize=17)
        plt.colorbar(im,cax=cax)
        plt.draw()
        
    def lorentz_wid(self, event):
        global marker
        marker = 5
        im = ax1.imshow(p_white, cmap='PiYG', vmin=-1, vmax=1)
        param = h_map[5]
        pflat = np.reshape(param, (param.shape[0]*param.shape[1]))
        pNaN = pflat[~np.isnan(pflat)]
        h_min = np.percentile(pNaN,1)  # set heatmap vmin to 1% of data (could lower to 0.5% or 0.1%)
        h_max = np.percentile(pNaN,99)  # set heatmap vmax to 99% of data (could up to 99.5% or 99.9%)
        #h_min = np.percentile(param,1)  # set heatmap vmin to 1% of data (could lower to 0.5% or 0.1%)
        #h_max = np.percentile(param,99)  # set heatmap vmax to 99% of data (could up to 99.5% or 99.9%)
        #h_min = 0.05
        #h_max = 0.5
        im = ax1.imshow(param, cmap='jet', interpolation='nearest', vmin=h_min, vmax=h_max,  picker=True)
        ax1.set_title(r'%s: %i $\AA$ | %s' % (date_title, wavelength, titles[5]), y = 1.01, fontsize=17)
        plt.colorbar(im,cax=cax)
        plt.draw()
        
    def fstat(self, event):
        global marker
        marker = 6
        param = h_map[6]
        NaN_replace = np.nan_to_num(param)  # NaN's in chi^2 heatmap were causing issue, replace with 0?
        #h_min = np.percentile(NaN_replace,1)  # set heatmap vmin to 1% of data (could lower to 0.5% or 0.1%)
        #h_max = np.percentile(NaN_replace,99)  # set heatmap vmax to 99% of data (could up to 99.5% or 99.9%)
        h_min = -10
        h_max = 2
        im = ax1.imshow(param, cmap='jet', interpolation='none', vmin=h_min, vmax=h_max,  picker=True)
        ax1.set_title(r'%s: %i $\AA$ | %s' % (date_title, wavelength, titles[6]), y = 1.01, fontsize=17)
        plt.colorbar(im,cax=cax)
        plt.draw()
        
    def visual(self, event):
        param = vis[0]
        h_min = np.percentile(param,1)  # set heatmap vmin to 1% of data (could lower to 0.5% or 0.1%)
        h_max = np.percentile(param,99)  # set heatmap vmax to 99% of data (could up to 99.5% or 99.9%)
        im = ax1.imshow(param, cmap='sdoaia%i' % wavelength, interpolation='nearest', vmin=h_min, vmax=h_max, picker=True)
        ax1.set_title(r'%s: %i $\AA$ | %s' % (date_title, wavelength, titles[7]), y = 1.01, fontsize=17)
        plt.colorbar(im,cax=cax)
        plt.draw()
    
        
    def mask(self, event):
        global toggle2
        global marker
        if toggle2 == 0:
            toggle2 = 1
            plotMask(marker)
        elif toggle2 == 1:
            toggle2 = 0
            plotMap(marker)              
        return toggle2
        
    def saveFig(self, event):
        global count

        outdir = 'C:/Users/Brendan/Desktop/Tool_Figures/'

        if not os.path.exists(os.path.dirname(outdir)):
            try:
                print ("Specified directory not found.")
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST: raise
        else:
            plt.savefig('%s%s_%i_%i.pdf' % (outdir, date, wavelength, count), bbox_inches='tight')
        count += 1
        return count
  

# Simple mouse click function to store coordinates
def onclick(event):
    global ix, iy, c
    ixx, iyy = event.xdata, event.ydata
    if ixx > 1. and iyy > 1.:
        ax2.clear()
        ax3.clear()
        del ax1.collections[:]
        plt.draw()
        print ('x = %d, y = %d' % ( ixx, iyy))  # print location of pixel
        ix = int(ixx)
        iy = int(iyy)
        
        s = np.zeros((spectra.shape[2]))
        s[:] = spectra[iy][ix][:]
        
        # assign equal weights to all parts of the curve
        df = np.log10(f_fit[1:len(f_fit)]) - np.log10(f_fit[0:len(f_fit)-1])
        df2 = np.zeros_like(f_fit)
        df2[0:len(df)] = df
        df2[len(df2)-1] = df2[len(df2)-2]
        
        if wavelength in [1600,1700]:
            ds = df2
        else:
            ds = stddev[iy][ix]
        
        ## fit data to combined power law plus gaussian component model
        
        try:
            # initial guesses for fitting parameters
            M1_low = [-0.002, 0.3, -0.01]
            M1_high = [0.002, 6., 0.01]
            nlfit_l, nlpcov_l = scipy.optimize.curve_fit(PowerLaw, f_fit, s, bounds=(M1_low, M1_high), sigma=ds, method='dogbox', max_nfev=3000)
          
        except RuntimeError:
            #print("Error M1 - curve_fit failed - %i, %i" % (l,m))  # turn off because would print too many to terminal
            print ("pl1")
            pass
        
        except ValueError:
            #print("Error M1 - inf/NaN - %i, %i" % (l,m))  # turn off because would print too many to terminal
            print ("pl2")
            pass
        
        A, n, C = nlfit_l  # unpack fitting parameters
         
         
        #"""        
        try:                                 
            M2_low = [0., 0.3, -0.01, 0.00001, -6.5, 0.05]
            M2_high = [0.002, 6., 0.01, 0.2, -4.6, 0.8]
            #M2_low = [0., 0.3, 0., 0.00001, 1., 0.]
            #M2_high = [0.002, 6., 0.01, 0.2, 100., 0.8]
        
                    
            # change method to 'dogbox' and increase max number of function evaluations to 3000
            #nlfit_gp, nlpcov_gp = scipy.optimize.curve_fit(LorentzPowerBase, f_fit, s, p0 = [A,n,0.,0.1,-5.55,0.425], bounds=(M2_low, M2_high), sigma=ds, method='dogbox', loss='huber',max_nfev=3000)
            nlfit_gp, nlpcov_gp = scipy.optimize.curve_fit(GaussPowerBase, f_fit, s, bounds=(M2_low, M2_high), sigma=ds, method='dogbox', max_nfev=3000)
            #nlfit_gp, nlpcov_gp = scipy.optimize.curve_fit(LorentzPowerBase, f, s, bounds=(M2_low, M2_high), sigma=ds, method='dogbox', ftol=.01, max_nfev=3000)
        
        except RuntimeError:
            #print("Error M2 - curve_fit failed - %i, %i" % (l,m))  # turn off because would print too many to terminal
            pass
        
        except ValueError:
            #print("Error M2 - inf/NaN - %i, %i" % (l,m))  # turn off because would print too many to terminal
            pass
        #"""
        
        A2, n2, C2, P2, fp2, fw2 = nlfit_gp  # unpack fitting parameters
               
        try:
            
            nlfit_gp2, nlpcov_gp2 = scipy.optimize.curve_fit(GaussPowerBase, f_fit, s, p0 = [A2, n2, C2, P2, fp2, fw2], bounds=(M2_low, M2_high), sigma=ds, max_nfev=3000)    
           
        except RuntimeError:
            #print("Error M2 - curve_fit failed - %i, %i" % (l,m))  # turn off because would print too many to terminal
            pass
        
        except ValueError:
            #print("Error M2 - inf/NaN - %i, %i" % (l,m))  # turn off because would print too many to terminal
            pass
        
        A22, n22, C22, P22, fp22, fw22 = nlfit_gp2  # unpack fitting parameters
        
        #print nlfit_gp2
        m2_param = nlfit_gp2
        #print('%0.3g %0.3g' % (A,A22))
        #print('%0.3f %0.3f' % (n,n22))
        #print('%0.3g %0.3g' % (C,C22))
        #print('%0.3e, %0.3f, %0.3e, %0.3e' % (A22, n22, C22, P22))
                       
        # create model functions from fitted parameters    
        m1_fit = PowerLaw(f_fit, A, n, C)  
        m1_fit2 = PowerLaw(f_fit, A22, n22, C22)  
        lorentz = Gauss(f_fit,P22,fp22,fw22)
        #m2_fit = LorentzPowerBase(f_fit, A2,n2,C2,P2,fp2,fw2)
        m2_fit2 = GaussPowerBase(f_fit, A22,n22,C22,P22,fp22,fw22) 
        #m2_fit = GaussPowerBase(f, 5.65e-7,1.49,1e-4,0.0156,-6.5,0.59)     
        
        residsM1 = (s - m1_fit)
        chisqrM1 =  ((residsM1/ds)**2).sum()
        redchisqrM1 = ((residsM1/ds)**2).sum()/float(f_fit.size-3)  
         
        #residsM2 = (s - m2_fit)
        #chisqrM2 = ((residsM2/ds)**2).sum()
        #redchisqrM2 = ((residsM2/ds)**2).sum()/float(f.size-6)
        
        residsM22 = (s - m2_fit2)
        chisqrM22 = ((residsM22/ds)**2).sum()
        redchisqrM22 = ((residsM22/ds)**2).sum()/float(f_fit.size-6) 
              
        #f_test = ((chisqrM1-chisqrM2)/(6-3))/((chisqrM2)/(f.size-6))
        f_test2 = ((chisqrM1-chisqrM22)/(6-3))/((chisqrM22)/(f_fit.size-6))
        
        df1, df2 = 3, 6  # degrees of freedom for model M1, M2
        p_val = ff.sf(f_test2, df1, df2)
        
        #amp_scale = PowerLaw(np.exp(fp2), A2, n2, C2)  # to extract the gaussian-amplitude scaling factor
        amp_scale2 = PowerLaw(np.exp(fp22), A22, n22, C22)  # to extract the gaussian-amplitude scaling factor
        
        r = pearsonr(m2_fit2, s)[0]  # calculate r-value correlation coefficient
        
        
        # calculate weighted correlation coefficient
        ds2 = ds
        #ds2 = stddev[iy][ix]
        weight_mean_spec = (ds2*s).sum()/ds2.sum()
        weight_mean_m2 = (ds2*m2_fit2).sum()/ds2.sum()
        weight_cov_spec_m2 = (ds2*(s-weight_mean_spec)*(m2_fit2-weight_mean_m2)).sum()/ds2.sum()
        weight_cov_spec = (ds2*(s-weight_mean_spec)*(s-weight_mean_spec)).sum()/ds2.sum()
        weight_cov_m2 = (ds2*(m2_fit2-weight_mean_m2)*(m2_fit2-weight_mean_m2)).sum()/ds2.sum()
        weight_corr = weight_cov_spec_m2/np.sqrt(weight_cov_spec*weight_cov_m2)
        
        chisqrM22B = ((residsM22/ds2)**2).sum()
        chisqrM1B =  ((residsM1/ds2)**2).sum()
        f_test2B = ((chisqrM1B-chisqrM22B)/(6-3))/((chisqrM22B)/(f_fit.size-6))
        #print(f_test2, f_test2B)
        
        
        plt.rcParams["font.family"] = "Times New Roman"
        font_size = 20
        
        ax2.set_title('Full M2', y = 1.01, fontsize=17) 
        ax2.loglog(f_fit, m1_fit, 'r--', linewidth=1.3, label='M1')
        ax2.loglog(f_fit, m1_fit2, 'g--', linewidth=1.3, label='M1 from M2')
        ax2.loglog(f_fit, m2_fit2, 'b', linewidth=1.3, label='M2 Combined')
        ax2.loglog(f_fit, lorentz, 'b--', linewidth=1.3, label='Kappa')
        ax2.loglog(f_fit, s, 'k', linewidth=1.3)
        ax2.loglog(f_fit, ds, 'r', label='Uncertainties')
        ax2.set_ylabel('Power', fontsize=font_size, labelpad=10, fontname="Times New Roman")
        ax2.axvline(x=0.00333,color='k',ls='dashed', label='5 minutes')
        ax2.axvline(x=0.00555,color='k',ls='dotted', label='3 minutes')
        #ax2.text(0.006, 10**-0.62, r'$\chi^2$ = {0:0.3f}'.format(chisqrM22), fontsize=font_size, fontname="Times New Roman")
        #ax2.text(0.006, 10**-0.98, r'$w_\chi$ = {0:0.3f}'.format(chisqrM22B), fontsize=font_size, fontname="Times New Roman")
        #ax2.text(0.007, 10**-1.34, r'$r$ = {0:0.3f}'.format(r), fontsize=font_size, fontname="Times New Roman")
        #ax2.text(0.006, 10**-1.70, r'$w_r$ = {0:0.3f}'.format(weight_corr), fontsize=font_size, fontname="Times New Roman")
        ax2.text(0.008, 10**-0.62, r'n = {0:0.2f}'.format(n22), fontsize=font_size, fontname="Times New Roman")
        ax2.text(0.008, 10**-1.06, r'$\beta$ = {0:0.2f}'.format((1./np.exp(fp22))/60.), fontsize=font_size, fontname="Times New Roman")
        ax2.text(0.006, 10**-1.50, r'FWHM = {0:0.2f}'.format((1./(np.exp(fp22+fw22)-np.exp(fp22-fw22)))/60.), fontsize=font_size, fontname="Times New Roman")
        #plt.vlines((0.0093),10**-8,10**1, linestyles='dotted', label='3 minutes')
        legend = ax2.legend(loc='lower left', prop={'size':15}, labelspacing=0.35)
        ax2.set_xlim(10**-4., 10**-1.3)
        ax2.set_ylim(10**-5, 10**0)   
        ax1.scatter(ix, iy, s=200, marker='x', c='white', linewidth=2.5)
        for label in legend.get_lines():
                label.set_linewidth(2.0)  # the legend line width   
                
        

        #ds = stddev[iy][ix]
        
        ## fit data to combined power law plus gaussian component model
        
        k = 80
        s2 = s[:k]
        f2 = f_fit[:k]
        ds2 = ds[:k]
        
        try:
            # initial guesses for fitting parameters
            M1_low = [-0.002, 0.3, -0.01]
            M1_high = [0.002, 6., 0.01]
            nlfit_l, nlpcov_l = scipy.optimize.curve_fit(PowerLaw, f2, s2, bounds=(M1_low, M1_high), sigma=ds2, method='dogbox')
          
        except RuntimeError:
            #print("Error M1 - curve_fit failed - %i, %i" % (l,m))  # turn off because would print too many to terminal
            pass
        
        except ValueError:
            #print("Error M1 - inf/NaN - %i, %i" % (l,m))  # turn off because would print too many to terminal
            pass
        
        A, n, C = nlfit_l  # unpack fitting parameters
         
        #"""        
        try:                                 
            M2_low = [0., 0.3, -0.01, 0.00001, -6.5, 0.05]
            M2_high = [0.002, 6., 0.01, 0.2, -4.6, 0.8]
            #M2_low = [0., 0.3, 0., 0.00001, 1., 0.]
            #M2_high = [0.002, 6., 0.01, 0.2, 100., 0.8]
        
                    
            # change method to 'dogbox' and increase max number of function evaluations to 3000
            #nlfit_gp, nlpcov_gp = scipy.optimize.curve_fit(LorentzPowerBase, f_fit, s, p0 = [A,n,0.,0.1,-5.55,0.425], bounds=(M2_low, M2_high), sigma=ds, method='dogbox', loss='huber',max_nfev=3000)
            nlfit_gp, nlpcov_gp = scipy.optimize.curve_fit(GaussPowerBase, f2, s2, bounds=(M2_low, M2_high), sigma=ds2, method='dogbox', max_nfev=3000)
            #nlfit_gp, nlpcov_gp = scipy.optimize.curve_fit(LorentzPowerBase, f, s, bounds=(M2_low, M2_high), sigma=ds, method='dogbox', ftol=.01, max_nfev=3000)
        
        except RuntimeError:
            #print("Error M2 - curve_fit failed - %i, %i" % (l,m))  # turn off because would print too many to terminal
            pass
        
        except ValueError:
            #print("Error M2 - inf/NaN - %i, %i" % (l,m))  # turn off because would print too many to terminal
            pass
        #"""
        
        A2, n2, C2, P2, fp2, fw2 = nlfit_gp  # unpack fitting parameters
               
        try:
            
            nlfit_gp2, nlpcov_gp2 = scipy.optimize.curve_fit(GaussPowerBase, f2, s2, p0 = [A2, n2, C2, P2, fp2, fw2], bounds=(M2_low, M2_high), sigma=ds2, max_nfev=3000)    
           
        except RuntimeError:
            #print("Error M2 - curve_fit failed - %i, %i" % (l,m))  # turn off because would print too many to terminal
            pass
        
        except ValueError:
            #print("Error M2 - inf/NaN - %i, %i" % (l,m))  # turn off because would print too many to terminal
            pass
        
        A22, n22, C22, P22, fp22, fw22 = nlfit_gp2  # unpack fitting parameters     
        #print nlfit_gp2
        #dA22, dn22, dC22, dP22, dfp22, dfw22 = [np.sqrt(nlpcov_gp[j,j]) for j in range(nlfit_gp.size)]
        
        #print('Index: %0.3f \nAmp.: %0.3e \nKappa: %0.2f \nrho: %0.2e \nrho_loc: %0.2f \nloc: %0.2f' % (nlfit_gp2[1], nlfit_gp2[3], kappa0, rho0, rho0*120*60, (1./np.exp(fp22))/60.)) 
        
        #m2_param = A22, n22, C22, P22, fp22, fw22  # could have used this for params array : = params[0:6,l-1,m-1]
        #uncertainties = dA22, dn22, dC22, dP22, dfp22, dfw22  # do we want to keep a global array of uncertainties?
                       
        # create model functions from fitted parameters    
        m1_fit = PowerLaw(f2, A22, n22, C22)    
        lorentz = Gauss(f2,P22,fp22,fw22)
        m2_fit2 = GaussPowerBase(f2, A22,n22,C22,P22,fp22,fw22)     
        
        residsM1 = (s2 - m1_fit)
        chisqrM1 =  ((residsM1/ds2)**2).sum()
        redchisqrM1 = ((residsM1/ds2)**2).sum()/float(f2.size-3)  
        
        residsM22 = (s2 - m2_fit2)
        chisqrM22 = ((residsM22/ds2)**2).sum()
        redchisqrM22 = ((residsM22/ds2)**2).sum()/float(f2.size-6) 
              
        #f_test = ((chisqrM1-chisqrM2)/(6-3))/((chisqrM2)/(f.size-6))
        f_test2 = ((chisqrM1-chisqrM22)/(6-3))/((chisqrM22)/(f2.size-6))
        
        df1, df2 = 3, 6  # degrees of freedom for model M1, M2
        p_val = ff.sf(f_test2, df1, df2)
        
        #amp_scale = PowerLaw(np.exp(fp2), A2, n2, C2)  # to extract the gaussian-amplitude scaling factor
        amp_scale2 = PowerLaw(np.exp(fp22), A22, n22, C22)  # to extract the gaussian-amplitude scaling factor
        
        ax3.set_title('Lorentzian', y = 1.01, fontsize=17)
        ax3.loglog(f2, m1_fit, 'r--', linewidth=1.3, label='M1')
        ax3.loglog(f2, m2_fit2, 'b', linewidth=1.3, label='M2 Combined')
        ax3.loglog(f2, lorentz, 'b--', linewidth=1.3, label='Lorentzian')
        ax3.loglog(f2, s2, 'k', linewidth=1.3)
        ax3.loglog(f2, ds2, 'r', label='Uncertainties')
        ax3.set_xlabel('Frequency [Hz]', fontsize=font_size, labelpad=10, fontname="Times New Roman")
        ax3.set_ylabel('Power', fontsize=font_size, labelpad=10, fontname="Times New Roman")
        ax3.axvline(x=0.00333,color='k',ls='dashed', label='5 minutes')
        ax3.axvline(x=0.00555,color='k',ls='dotted', label='3 minutes')
        #ax3.text(0.006, 10**-0.62, r'$\chi^2$ = {0:0.3f}'.format(chisqrM22), fontsize=font_size, fontname="Times New Roman")
        #ax3.text(0.007, 10**-1.06, r'$r$ = {0:0.3f}'.format(r), fontsize=font_size, fontname="Times New Roman")
        #ax3.text(0.006, 10**-1.50, r'$w_r$ = {0:0.3f}'.format(weight_corr), fontsize=font_size, fontname="Times New Roman")
        ax3.text(0.008, 10**-0.62, r'n = {0:0.2f}'.format(n22), fontsize=font_size, fontname="Times New Roman")
        ax3.text(0.008, 10**-1.06, r'$\beta$ = {0:0.2f}'.format((1./np.exp(fp22))/60.), fontsize=font_size, fontname="Times New Roman")
        ax3.text(0.006, 10**-1.50, r'FWHM = {0:0.2f}'.format((1./(np.exp(fp22+fw22)-np.exp(fp22-fw22)))/60.), fontsize=font_size, fontname="Times New Roman")
        #plt.vlines((0.0093),10**-8,10**1, linestyles='dotted', label='3 minutes')
        legend = ax3.legend(loc='lower left', prop={'size':15}, labelspacing=0.35)
        ax3.set_xlim(10**-4., 10**-1.3)
        ax3.set_ylim(10**-5, 10**0)   
        for label in legend.get_lines():
                label.set_linewidth(2.0)  # the legend line width   
        #print(m1_fit[0], lorentz[0], m2_fit2[0])
        print(1./f2[0], 1./f2[-1])

    return ix, iy
    
# define combined-fitting function (Model M2)
def LorentzPowerBase(f2, A2, n2, C2, P2, fp2, fw2):
    #return A2*f2**-n2 + C2 + P2*(1./ (1.+((np.log(f2)-fp2)/fw2)**2))
    return A2*f2**-n2 + C2 + P2*((1 + (f2**2 / (fp2*fw2**2)))**(-(fp2+1)/2))  # kappa 
             
# define combined-fitting function (Model M2)
def GaussPowerBase(f2, A2, n2, C2, P2, fp2, fw2):
    #return A2*f2**-n2 + C2 + P2*np.exp(-0.5*(((np.log(f2))-fp2)/fw2)**2)  
    return A2*f2**-n2 + C2 + P2*(1./ (1.+((np.log(f2)-fp2)/fw2)**2))  # lorentz

# define Power-Law-fitting function (Model M1)
def PowerLaw(f, A, n, C):
    return A*f**-n + C
        
# define Gaussian-fitting function
def Lorentz(f2, P2, fp2, fw2):
    #return P*(1./ (1.+((np.log(f)-fp)/fw)**2))
    return P2*((1 + (f2**2 / (fp2*fw2**2)))**(-(fp2+1)/2))
    
# define Gaussian-fitting function
def Gauss(f, P, fp, fw):
    #return P*np.exp(-0.5*(((np.log(f))-fp)/fw)**2) 
    return P*(1./ (1.+((np.log(f)-fp)/fw)**2))
    

"""
##############################################################################
##############################################################################
"""

directory = 'F:'
#date = '20001111'
#date = '20130626'
date = '20140818'
#wavelength = 1700
wavelength = 1600

global spectra
global param1
global stddev

cube_shape = np.load('%s/DATA/Temp/%s/%i/spectra_mmap_shape.npy' % (directory, date, wavelength))
spectra = np.memmap('%s/DATA/Temp/%s/%i/spectra_mmap.npy' % (directory, date, wavelength), dtype='float64', mode='r', shape=(cube_shape[0], cube_shape[1], cube_shape[2]))
#stddev = np.memmap('%s/DATA/Temp/%s/%i/uncertainties_mmap.npy' % (directory, date, wavelength), dtype='float64', mode='r', shape=(cube_shape[0], cube_shape[1], cube_shape[2]))
#stddev = np.memmap('%s/DATA/Temp/%s/%i/3x3_stddev_mmap.npy' % (directory, date, wavelength), dtype='float64', mode='r', shape=(cube_shape[0], cube_shape[1], cube_shape[2]))

param1 = np.load('%s/DATA/Output/%s/%i/param.npy' % (directory, date, wavelength))


global marker
global toggle
global toggle2
global count
marker = 1
toggle = 0
toggle2 = 0
count = 0


### determine frequency values that FFT will evaluate
if wavelength == 1600 or wavelength == 1700:
    time_step = 24
else:
    time_step = 12
freq_size = (cube_shape[2]*2)+1
sample_freq = fftpack.fftfreq(freq_size, d=time_step)
pidxs = np.where(sample_freq > 0)    


if 1:
    
    global f_fit
    
    freqs = sample_freq[pidxs]
    print (len(freqs))
    f_fit = np.linspace(freqs[0],freqs[len(freqs)-1],int(spectra.shape[2]))   
    
    
    h_map = np.load('%s/DATA/Output/%s/%i/param.npy' % (directory, date, wavelength))
    
    p_white = np.zeros_like(h_map[0])
 
    vis = np.load('%s/DATA/Output/%s/%i/visual.npy' % (directory, date, wavelength))

    
    # arrays containing interesting points to be clicked for each dataset
    if date == '20120923' and wavelength == 211:
        x = [250, 359, 567, 357, 322, 315, 97, 511, 316, 336]  
        y = [234, 308, 218, 197, 201, 199, 267, 5, 175, 181]
    
    # create list of titles and colorbar names for display on the figures
    titles = ['Power Law Slope Coeff.', 'Power Law Index', 'Rollover [min]', 'Lorentzian Amplitude', 'Lorentzian Location [min]', 'Lorentzian Width', 'F-Statistic', 'Averaged Visual Image']
    date_title = '%i/%02i/%02i' % (int(date[0:4]),int(date[4:6]),int(date[6:8]))
    
    # create figure with heatmap and spectra side-by-side subplots
    fig1 = plt.figure(figsize=(18,12))
    ax1 = plt.gca()
    ax1 = plt.subplot2grid((30,28),(4, 1), colspan=14, rowspan=25)

    ax1.set_xlim(0, h_map.shape[2]-1)
    ax1.set_ylim(0, h_map.shape[1]-1)  
    ax1.set_title(r'%s: %i $\AA$ | %s' % (date_title, wavelength, titles[1]), y = 1.01, fontsize=17)
    
    # was getting error "'AxesImage' object is not iterable"
    # - found: "Each element in img needs to be a sequence of artists, not a single artist."
    param = h_map[1]  # set initial heatmap to power law index     
    h_min = np.percentile(param,1)  # set heatmap vmin to 1% of data (could lower to 0.5% or 0.1%)
    h_max = np.percentile(param,99)  # set heatmap vmax to 99% of data (could up to 99.5% or 99.9%)
    im, = ([ax1.imshow(param, cmap='jet', interpolation='nearest', vmin=h_min, vmax=h_max,  picker=True)])
    
    # design colorbar for heatmaps
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="3%", pad=0.07)
    cbar = plt.colorbar(im,cax=cax)
    #cbar.set_label('%s' % cbar_labels[1], size=15, labelpad=10)
    cbar.ax.tick_params(labelsize=13, pad=3)   
    
    
    # make toggle buttons to display each parameter's heatmap
    axindex = plt.axes([0.01, 0.9, 0.05, 0.063])
    axroll = plt.axes([0.07, 0.9, 0.05, 0.063])
    axlorentz_amp = plt.axes([0.13, 0.9, 0.05, 0.063])
    axlorentz_loc = plt.axes([0.19, 0.9, 0.05, 0.063])
    axlorentz_wid = plt.axes([0.25, 0.9, 0.05, 0.063])
    axfstat = plt.axes([0.31, 0.9, 0.05, 0.063])
    axvisual = plt.axes([0.37, 0.9, 0.05, 0.063])
    axmask = plt.axes([0.43, 0.9, 0.05, 0.063])
    axsaveFig = plt.axes([0.49, 0.9, 0.05, 0.063])
 
    # set up spectra subplot
    ax2 = plt.subplot2grid((30,28),(1, 17), colspan=10, rowspan=14)
    ax2.loglog()
    ax2.set_xlim(10**-4., 10**-1.3)
    ax2.set_ylim(10**-5, 10**0)  
    
    fig1.canvas.mpl_connect('button_press_event', onclick)
    
    ax2.set_title('Weightings: Frequency Spacing', y = 1.01, fontsize=17)
    
    ax3 = plt.subplot2grid((30,28),(15, 17), colspan=10, rowspan=14)
    ax3.loglog()
    ax3.set_xlim(10**-4., 10**-1.3)
    ax3.set_ylim(10**-5, 10**0)  
    ax3.set_title('Weightings: 3x3 Std. Dev.', y = 1.01, fontsize=17)
    
    plt.tight_layout()
    
    
    # add callbacks to each button - linking corresponding action
    callback = Index()
    
    bindex = Button(axindex, 'Index')
    bindex.on_clicked(callback.index)
    broll = Button(axroll, 'Rollover')
    broll.on_clicked(callback.roll)
    blorentz_amp = Button(axlorentz_amp, 'Lorentz Amp')
    blorentz_amp.on_clicked(callback.lorentz_amp)
    blorentz_loc = Button(axlorentz_loc, 'Lorentz Loc')
    blorentz_loc.on_clicked(callback.lorentz_loc)
    blorentz_wid = Button(axlorentz_wid, 'Lorentz Wid')
    blorentz_wid.on_clicked(callback.lorentz_wid)
    bfstat = Button(axfstat, 'F-Stat')
    bfstat.on_clicked(callback.fstat)
    bvisual = Button(axvisual, 'Visual')
    bvisual.on_clicked(callback.visual)
    bmask = Button(axmask, 'Mask')
    bmask.on_clicked(callback.mask)
    bsaveFig = Button(axsaveFig, 'Save')
    bsaveFig.on_clicked(callback.saveFig)
    
plt.draw()