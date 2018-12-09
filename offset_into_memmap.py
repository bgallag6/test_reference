# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 10:33:02 2018

@author: Brendan
"""

import numpy as np

"""
r = np.arange(60)
arr = r.reshape((3,4,5))

orig_shape = np.array([3,4,5])
        
# create memory-mapped array with similar datatype and shape to original array
mmap_arr = np.memmap('mmapTest2', dtype='int16', mode='w+', shape=tuple(orig_shape))

# write data to memory-mapped array
mmap_arr[:] = arr[:]

# flush memory changes to disk, then remove memory-mapped object and original array
del mmap_arr
"""

"""
orig_shape = np.array([3,4,5])
#off_set = 1*4*5*(32//8)
row = 1
col = 2 
off_set = ((row)*4*5 + (col)*5)*(32//16)
arr2 = np.memmap('mmapTest2', dtype='int16', mode='r', shape=(5),
                 offset=off_set)
"""




"""
r = np.arange(60)
arr = r.reshape((5,4,3))

orig_shape = np.array([5,4,3])
        
# create memory-mapped array with similar datatype and shape to original array
mmap_arr = np.memmap('mmapTest3', dtype='int16', mode='w+', shape=tuple(orig_shape))

# write data to memory-mapped array
mmap_arr[:] = arr[:]

# flush memory changes to disk, then remove memory-mapped object and original array
del mmap_arr
"""

#"""
orig_shape = np.array([5,4,3])
#off_set = 1*4*5*(32//8)
row = 1
col = 2 
off_set = ((row)*4*5 + (col)*5)*(32//16)
arr2 = np.memmap('mmapTest2', dtype='int16', mode='r', shape=(5),
                 offset=off_set)
#"""
