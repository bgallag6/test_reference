# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 20:09:31 2018

@author: Brendan
"""

import sys
from mpi4py import MPI

def func1(x):
    for i in range(x):
        if i == 9:
            if rank == 0:
                sys.exit("Error: please use python 3")
            else:
                sys.exit()
        print(i, flush=True)





h = 12
       
comm = MPI.COMM_WORLD   # Set up comms
rank = comm.Get_rank()  # Each processor gets its own "rank"
size = comm.Get_size()

z = 3+h
            
     
func1(z)