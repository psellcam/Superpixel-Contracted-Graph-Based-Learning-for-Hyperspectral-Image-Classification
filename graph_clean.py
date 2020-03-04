#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 13:25:16 2019

@author: ps644
"""
import graph
import numpy as np
import math
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import csr_matrix
import scipy
import time 
## Class for graph based algorithms 
class graph_semi_supervised(object):   
    
     # Initialisation function
    def __init__(self,weight_graph,labels):      
        
        # 2D Array of weights
        if(np.amin(weight_graph) < 0):
            raise ValueError('Negative weights inputed')
        else:
            self.weight_matrix = csr_matrix(weight_graph)
        
        # Inputted labels. Unlabeled points should be -1 , classes ranging from 0 up 
        if(len(labels.shape) == 1):
            self.class_labels = labels
            self.number_of_classes = np.amax(labels)+1
        
        if(len(labels.shape) > 1):
            self.class_labels = labels
            self.number_of_classes = labels.shape[1]
            
        self.node_number = self.weight_matrix.shape[0]
        self.y_matrix_creation()
        
        # Stores degrees of nodes in array
        self.degrees = csr_matrix.sum(self.weight_matrix,axis = 1)
        self.degrees= np.asarray(self.degrees).reshape(-1)
        for i in range(self.node_number):
            if(self.degrees[i] == 0):
                self.degrees[i] += np.nextafter(np.float32(0), np.float32(1))
        
    # Change class labels into Y matrix and Y original and creates fidelity
    def y_matrix_creation(self):
        
        if(len(self.class_labels.shape) == 1):            
            self.Y = np.zeros((self.class_labels.shape[0],self.number_of_classes))
            self.F = np.zeros((self.class_labels.shape[0]))
            for i in range(self.class_labels.shape[0]):
                if(self.class_labels[i] > -1):
                    self.Y[i,self.class_labels[i]] = 1 
                    self.F[i] = 1
            
        if(len(self.class_labels.shape) > 1):      
            self.Y = np.copy(self.class_labels)
            self.F = np.zeros((self.class_labels.shape[0]))
            for i in range(self.class_labels.shape[0]):
                if(np.amax(self.class_labels[i,:]) > 0):
                    self.F[i] = 1
            
            
    # Function which implements local and global consistency
    def lgc_solver(self,mu):       
        d_neg_half = np.diag(np.power(self.degrees,-0.5))
        d_neg_half = csr_matrix(d_neg_half)
        step = d_neg_half.dot(self.weight_matrix)
        S =  (1/(1+mu)) *  step.dot(d_neg_half)
        S = scipy.sparse.identity(self.node_number) - S
        S = csr_matrix.todense(S)
        S = np.linalg.inv(S)     
        S = (mu/(1+mu)) * S 
        
        output_labels = np.linalg.multi_dot((S,self.Y))
        
        return output_labels
    
    

def lgc_classifier_m(mark_2,node_labels,mu_i,mode):
    graph_solver = graph_semi_supervised(mark_2,node_labels)
    lgc_probs = graph_solver.lgc_solver(mu = mu_i)   
    if(mode == 0):
     for i in range(lgc_probs.shape[0]):
         lgc_probs[i,:] = lgc_probs[i,:] / np.sum(lgc_probs[i,:])
    if(mode == 1):
        lgc_probs = np.argmax(lgc_probs,axis =1) + 1 
    return lgc_probs


def weighted_graph_m(feature_vector,sigma_i,neighbours):
    ## Create the weight matrix for the weighted features
    weight_matrix = kneighbors_graph(feature_vector, feature_vector.shape[0], mode='distance', include_self=True).toarray()
    weight_matrix = np.square(weight_matrix)    
    weight_matrix = np.exp(-weight_matrix/(sigma_i*sigma_i))
    np.fill_diagonal(weight_matrix,0)    
    return weight_matrix



def fcf_m(sigma,mu,k,node_labels,feature_vector):        
    weight_matrix = weighted_graph_m(feature_vector,sigma,feature_vector.shape[0]-1)     
    neighbours = k
    # %%
    for i in range(feature_vector.shape[0]):
        test = weight_matrix[i,:]
        s_test = np.sort(test)        
        test[np.where(test < s_test[-neighbours])] = 0
    # %%
    for i in range(feature_vector.shape[0]):
        for j in range(i+1,feature_vector.shape[0]):
            if(weight_matrix[i,j] != weight_matrix[j,i]):
                weight_matrix[i,j] = weight_matrix[j,i] = np.maximum(weight_matrix[i,j],weight_matrix[j,i])
    # %%    
    start = time.time()
    v_2_labels= lgc_classifier_m(weight_matrix,node_labels,mu,1)    
    final = time.time()
   
    # %%   
    return v_2_labels