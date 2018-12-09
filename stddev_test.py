# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 07:13:57 2018

@author: Brendan
"""

import numpy as np

# test to see if averaging std. dev. of segments, then of pixel-box, is equal to std. dev. of all at the end

x = np.random.random(200)
y = np.random.random(200)

x0 = [x[:100], x[100:]]
y0 = [y[:100], y[100:]]

avg_stdx = np.std(x0, axis=0)
avg_stdy = np.std(y0, axis=0)
#avg_stdx = (np.std(x[:100]) + np.std(x[100:])) / 2
#avg_stdy = (np.std(y[:100]) + np.std(y[100:])) / 2

xy = [avg_stdx, avg_stdy]

avg_stdc = np.std(xy, axis=0)
#avg_stdc = (avg_stdx + avg_stdy) / 2

xyfull = [x0,y0]
avg_stdf0 = np.std(xyfull, axis=1)
avg_stdf = np.std(avg_stdf0, axis=0)

diff = avg_stdf - avg_stdc

print(diff.max(), diff.min())


# both ways give the same result.  so I can either save a separate array, or save all the spectra segments