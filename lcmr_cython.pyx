#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 17:16:03 2019

@author: ps644
"""


import numpy as np
cimport numpy as np
cimport cython

from cython.parallel cimport prange
from libc.math cimport log
from libc.math cimport sqrt
from libc.math cimport pow
from libc.math cimport round
from libc.stdio cimport printf

@cython.boundscheck(False)
def norm_creation(double [:,:,::1] window_matrix):
    
    cdef double[:,::1] norms_2d  = np.zeros((window_matrix.shape[0]*window_matrix.shape[0],window_matrix.shape[2]))
    cdef double[:,::1] window_2d  = np.zeros((window_matrix.shape[0]*window_matrix.shape[0],window_matrix.shape[2]))
    cdef double[:,::1] final_2d = np.zeros((window_matrix.shape[0]*window_matrix.shape[0],window_matrix.shape[2]))
    cdef int i,j,k,window_dim,band_dim,count
    
    window_dim = window_matrix.shape[0]
    band_dim = window_matrix.shape[2]
    
    with nogil:
        for k in range(band_dim):
            count = 0
            for j in range(window_dim):
                for i in range(window_dim):
                    window_2d[count,k] = window_matrix[i,j,k]
                    count += 1
                    
        for i in range(window_dim*window_dim):
            for j in range(band_dim):
                    norms_2d[i,0] += pow(window_2d[i,j],2)
        
        for i in range(window_dim*window_dim):
            norms_2d[i,0] = sqrt(norms_2d[i,0])
            
        for i in range(window_dim*window_dim):
            for j in range(band_dim):
                    norms_2d[i,j] = norms_2d[i,0]
                    
        for i in range(window_dim*window_dim):
            for j in range(band_dim):
                final_2d[i,j] = window_2d[i,j] / norms_2d[i,j]
                    
    return  final_2d, window_2d
                    
        

@cython.boundscheck(False)
@cython.wraparound(False)
### Function that create the covariance matrices for clustering
def create_logm_matrices(np.ndarray[np.float64_t, ndim=3] spectral_mnf, int window_size, int K):    
    ## Variable List
    ## spectral_mnf: The reduced image
    ## window_size: the window size focused on each pixel
    ## K: The number of neighbours
    
    ## Output List
    ## all_matrices: The lcmr matrices in the shape dxdxhxw 
    
    cdef double tol
    cdef int size_0,size_1,d,i_d,i,pp,scale,j
    
    
    tol = 0.001
    size_0 = spectral_mnf.shape[0]
    size_1 = spectral_mnf.shape[1]
    d = spectral_mnf.shape[2]
    scale = int(np.floor(window_size/2))
    i_d = int(np.ceil(window_size*window_size/2))  
    
    
    
      
    cdef np.ndarray[np.float64_t, ndim=3] spectral_extended = np.pad(spectral_mnf,((scale,scale),(scale,scale),(0,0)),'symmetric')    
    
    cdef int padded_1 = spectral_extended.shape[1]   
    cdef np.ndarray[np.float64_t, ndim=3] tt_idea = np.zeros([window_size,padded_1,d])  
    cdef np.ndarray[np.float64_t, ndim=1] tt_norms_big = np.zeros([window_size*padded_1])    
    cdef np.ndarray[np.float64_t, ndim=2] tt_norms_big_repmat = np.zeros([window_size*padded_1,d])  
    cdef np.ndarray[np.float64_t, ndim=2] big_divide = np.zeros([window_size*padded_1,d])  
    cdef np.ndarray[np.float64_t, ndim=2] tt_temp_2d = np.zeros([window_size*window_size,d])  
    cdef np.ndarray[np.float64_t, ndim=2] norm_2d = np.zeros([window_size*window_size,d])  
    cdef np.ndarray[np.float64_t, ndim=1] center_pixel = np.zeros([d]) 
    cdef np.ndarray[np.float64_t, ndim=1] cor = np.zeros([window_size*window_size]) 
    cdef np.ndarray[np.int64_t, ndim=1] indexes = np.zeros([K],dtype=np.int64) 
    cdef np.ndarray[np.float64_t, ndim=2] tmp_mat = np.zeros([K,d])  
    cdef np.ndarray[np.float64_t, ndim=1] m = np.zeros([K]) 
    cdef np.ndarray[np.float64_t, ndim=2] m_matrix = np.zeros([K,d])  
    cdef np.ndarray[np.float64_t, ndim=1] M = np.zeros([K]) 
    cdef np.ndarray[np.float64_t, ndim=2] M_matrix = np.zeros([K,d])  
    cdef np.ndarray[np.float64_t, ndim=2] scaled_matrix = np.zeros([K,d])  
    cdef np.ndarray[np.float64_t, ndim=1] mean_scaled_matrix = np.zeros([d])  
    cdef np.ndarray[np.float64_t, ndim=2] centered_mat = np.zeros([K,d])  
    cdef np.ndarray[np.float64_t, ndim=2] tmp = np.zeros([d,d]) 
    #cdef np.ndarray[np.float64_t, ndim=2] tmp_tmp = np.zeros([d,d])  
    cdef np.ndarray[np.float64_t, ndim=1] s = np.zeros([d])
    cdef np.ndarray[np.float64_t, ndim=2] u = np.zeros([d,d])  
    cdef np.ndarray[np.float64_t, ndim=2] hmm = np.zeros([d,d])  
    cdef np.ndarray[np.float64_t, ndim=4] all_matrices = np.zeros([d,d,size_0,size_1])   
    
    
    for i in range(size_0):
        pp = i+(2*scale)+1
        tt_idea = spectral_extended[i:pp,:,:] 
        tt_big_2d = np.reshape(tt_idea ,(window_size*tt_idea.shape[1],d),order='F')
        tt_norms_big = np.linalg.norm(tt_big_2d,axis=1)
        tt_norms_big_repmat = np.transpose(np.matlib.repmat(tt_norms_big,d,1))
        big_divide = np.divide(tt_big_2d,tt_norms_big_repmat)
        #big_divide = np.ascontiguousarray(np.divide(tt_big_2d,tt_norms_big_repmat))
       
        for j in range(size_1):               
# %%           
            tt_temp_2d = tt_big_2d[(j*window_size):((window_size*window_size)+j*window_size),:]
            norm_2d = big_divide[(j*window_size):((window_size*window_size)+j*window_size),:]
# %%
            center_pixel = norm_2d[i_d-1,:]
            cor = np.linalg.multi_dot((norm_2d,center_pixel)) 
            indexes = (-cor).argsort()[:K]            
            tmp_mat = tt_temp_2d[indexes,:]
            m = np.amin(tmp_mat,axis=1)
            m_matrix = np.transpose(np.matlib.repmat(m,d,1))              
            M = np.amax(tmp_mat,axis=1)
            M_matrix = np.transpose(np.matlib.repmat(M-m,d,1))                
            scaled_matrix = np.divide((tmp_mat - m_matrix),M_matrix)
            mean_scaled_matrix = np.average(scaled_matrix,axis=0)            
            centered_mat = scaled_matrix - np.matlib.repmat(mean_scaled_matrix,tmp_mat.shape[0],1)   
          
            tmp = np.dot(np.transpose(centered_mat),centered_mat) 
            tmp = np.divide(tmp,K-1) 
            tmp_tmp = tmp + tol*np.identity(d)*np.trace(tmp) 
            [s,u] = np.linalg.eigh(tmp_tmp)     
            hmm = np.diag(np.log(s))
            all_matrices[:,:,i,j]  = np.linalg.multi_dot((u,hmm,np.linalg.inv(u)))   
            
    return all_matrices



@cython.boundscheck(False)
@cython.wraparound(False)
def cylog(np.ndarray[double, ndim=2] a):
    
    out = np.empty((a.shape[0],a.shape[1]), dtype=a.dtype)
    cdef double [:, :] out_view = out
    
    cdef Py_ssize_t i,j
    cdef double store
    with nogil:
        for i in prange(a.shape[0]):
            for j in prange(a.shape[1]):
                out_view[i,j] =log(a[i,j])
    return out




