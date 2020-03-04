"""
Created on Mon Feb 12 09:48:08 2018

@author: ps644
"""
cimport cython
import numpy as np
from libc.math cimport sqrt
from libc.math cimport round
from libc.stdio cimport printf

@cython.boundscheck(False)
def create_distance_matrix(double[:,::1] cluster_positions):
    
    # get dimensions
    cdef Py_ssize_t cluster_number
    cluster_number = cluster_positions.shape[0]

    # neighborhood arrays
    cdef double[:,::1] distances = np.zeros((cluster_number,cluster_number))    
    cdef double x_d , y_d
    cdef Py_ssize_t i,j
    cdef double distance_mag 
  
    with nogil:
        for i in range(cluster_number):            
            for j in range(i+1,cluster_number):
                distance_mag = 0
                x_d = cluster_positions[i,0] - cluster_positions[j,0]  
                y_d = cluster_positions[i,1] - cluster_positions[j,1] 
                
                distance_mag += x_d * x_d
                distance_mag += y_d * y_d
                distances[i,j] = distances[j,i] = sqrt(distance_mag)
                        
    return distances
                        