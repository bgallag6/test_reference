# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 09:39:06 2018

@author: Brendan
"""

import numpy as np
import matplotlib.pyplot as plt

directory = "C:/Users/Brendan/Desktop/specFit/images/processed/20120606/1600"

#cube = np.load('%s/dataCube.npy' % directory)
#vis = np.load('%s/visual.npy' % directory)
#timestamps = np.load('%s/timestamps.npy' % directory)
#exposures = np.load('%s/exposures.npy' % directory)
params = np.load('%s/param.npy' % directory)
specCube = np.load('%s/specCube.npy' % directory)
frequencies = np.load('%s/frequencies.npy' % directory)
vis = np.load('%s/visual.npy' % directory)


directory = "C:/Users/Brendan/Desktop/specFit/images/validation/processed/20120606/1600"

params2 = np.load('%s/param.npy' % directory)
specCube2 = np.load('%s/specCube.npy' % directory)
frequencies2 = np.load('%s/frequencies.npy' % directory)
vis2 = np.load('%s/visual.npy' % directory)

print(np.max(params-params2), np.min(params-params2))
print(np.max(specCube-specCube2), np.min(specCube-specCube2))
print(np.max(frequencies-frequencies2), np.min(frequencies-frequencies2))
print(np.max(vis-vis2), np.min(vis-vis2))

#t = np.array([i for i in range(256)])

#np.save('%s/timestamps.npy' % directory, t)



"""
def M2(f, a, n, c):
    return a*f**-n + c

m2 = M2(frequencies, *params[:3,0,0])

plt.figure()
#plt.plot(timestamps, cube[:,0,0])
plt.loglog(frequencies, specCube[0,0])
plt.loglog(frequencies, m2)
"""


