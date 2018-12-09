# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 08:40:16 2018

@author: Brendan
"""

import numpy as np
import matplotlib.pyplot as plt
#import numba  # speed-up ~10%?

# define Power-Law-fitting function (Model M1)
#@numba.jit
#def M1(f, a, n, c):
#    return a*f**-n + c
def M1(f, p0, p1, p2):
    return p0*f**-p1 + p2

def M2(f, p0, p1):
    return p0*f**-p1

"""
# define combined-fitting function (Model M2)
#@numba.jit
def M2(f, a, n, c, p, fp, fw):
    return a*f**-n + c + p*(1./ (1.+((np.log(f)-fp)/fw)**2))

# define Lorentzian-fitting function
#@numba.jit
def m2(f, p, fp, fw):
    return p*(1./ (1.+((np.log(f)-fp)/fw)**2))
"""

"""
freqs = np.linspace(10**-5, 10**-2, 299)

pM1 = (1e-7, 1.5, 1e-3)

last_m1 = M1(freqs, *pM1)
noisy_m1 = last_m1 + np.random.random(299)*last_m1



from scipy.optimize import curve_fit as Fit

m1_param = Fit(M1, freqs, noisy_m1, p0=[1e-7, 1.5, 1e-4])[0]
                  
m1_fit = M1(freqs, *m1_param)   

m2_fit = M2(freqs, *m1_param[:2])

plt.figure()
plt.loglog(freqs, noisy_m1)
plt.loglog(freqs, m1_fit)
plt.loglog(freqs, m2_fit)

params = np.zeros((2,2,6))

params[0][0][:3] = m1_param
"""

x = 'a*f**-n + c + p*(1./ (1.+((np.log(f)-fp)/fw)**2))'
string = 'p0*f**-p1 + p2 + p3(1./ (1.+((np.log(f)-p4)/p5)**2))'

f = freqs
a = 1e-8
n = 1.5
c = 1e-4
p = 1e-3
fp = -5.
fw = 0.12

y = eval(x)

plt.figure()
plt.loglog(f, y)
