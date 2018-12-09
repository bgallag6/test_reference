# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 10:41:25 2018

@author: Brendan
"""

import numba
import numpy as np
from scipy.optimize import curve_fit as Fit
from timeit import default_timer as timer
from scipy import fftpack


sample_freq = fftpack.fftfreq(300, d=24)
pidxs = np.where(sample_freq > 0)    
freqs = sample_freq[pidxs]

# assign equal weights to all parts of curve & use as fitting uncertainties
df = np.log10(freqs[1:len(freqs)]) - np.log10(freqs[0:len(freqs)-1])
df2 = np.zeros_like(freqs)
df2[0:len(df)] = df
df2[len(df2)-1] = df2[len(df2)-2]
ds = df2




def M2(f2, A2, n2, C2, P2, fp2, fw2):
    return A2*f2**-n2 + C2 + P2*(1./ (1.+((np.log(f2)-fp2)/fw2)**2))    

@numba.jit
def M2_jit(f2, A2, n2, C2, P2, fp2, fw2):
    return A2*f2**-n2 + C2 + P2*(1./ (1.+((np.log(f2)-fp2)/fw2)**2))    


M2_low = [0., 0.3, -0.01, 0.00001, -6.5, 0.05]
M2_high = [0.002, 6., 0.01, 0.2, -4.6, 0.8]
            
f = freqs

k = 200
s = np.array([M2(freqs, 1e-7, 1.3, 1e-4, 1e-3, -5.1, 0.12) for i in range(k)])

for i in range(k):
    s[i] = s[i] + np.random.randn(len(freqs))*0.2*s[i]



@numba.jit
def fit_ex(s):
    nlfit_gp, nlpcov_gp = Fit(M2_jit, f, s, bounds=(M2_low, M2_high), 
                              sigma=ds, method='dogbox', max_nfev=3000)
    return nlfit_gp
        
a = np.zeros((k,6), dtype=np.float64)

start = timer()
for i in range(k):
    a[i] = fit_ex(s[i])
end = timer()

print(end-start)



a3 = np.zeros((k,6), dtype=np.float64)
start = timer()
for i in range(k):
    nlfit_gp, nlpcov_gp = Fit(M2_jit, f, s[i], bounds=(M2_low, M2_high), 
                              sigma=ds, method='dogbox', max_nfev=3000)
    a3[i] = nlfit_gp

end = timer()

print(end-start)




a2 = np.zeros((k,6), dtype=np.float64)
start = timer()
for i in range(k):
    nlfit_gp, nlpcov_gp = Fit(M2, f, s[i], bounds=(M2_low, M2_high), 
                              sigma=ds, method='dogbox', max_nfev=3000)
    a2[i] = nlfit_gp

end = timer()

print(end-start)

diff1 = a2-a
diff2 = a3-a
print(diff1.max(), diff1.min(), diff2.max(), diff2.min())
