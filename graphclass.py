#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 09:12:31 2018

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
    
    
    # Function which implements local and global consistency
    def LS_solver(self,alpha):       
        d_neg_half = np.diag(np.power(self.degrees,-0.5))
        S = np.linalg.multi_dot((d_neg_half,self.weight_matrix,d_neg_half))
        S = np.identity(self.node_number) - alpha*S
        S = np.linalg.inv(S)     
        
        output_labels = np.linalg.multi_dot((S,self.Y))
        
        return output_labels

                
def mean_weighted_all_graph(mslic,weighted_feature_vector,mean_feature_vector,sigma_i,b_i):
    ## Create the weight matrix for the weighted features
    weight_matrix_w = kneighbors_graph(weighted_feature_vector, mslic.ClusterNumber, mode='distance', include_self=True).toarray()
    weight_matrix_w = np.square(weight_matrix_w)    
    ## Create the weight matrix for mean features
    weight_matrix_m = kneighbors_graph(mean_feature_vector, mslic.ClusterNumber, mode='distance', include_self=True).toarray()
    weight_matrix_m = np.square(weight_matrix_m)
    
    weight_matrix = b_i * weight_matrix_w + [1-b_i] * weight_matrix_m

    weight_matrix = np.exp(-weight_matrix/(sigma_i*sigma_i))
    np.fill_diagonal(weight_matrix,0)
    
    return weight_matrix

def mean_weighted_k_graph(mslic,weighted_feature_vector,mean_feature_vector,sigma_i,b_i,neighbours):    
    ## Create the weight matrix for weight features
    weight_matrix_w = kneighbors_graph(weighted_feature_vector, mslic.ClusterNumber, mode='distance', include_self=True).toarray()
    weight_matrix_w = np.square(weight_matrix_w)    
    ## Create the weight matrix for mean features
    weight_matrix_m = kneighbors_graph(mean_feature_vector, mslic.ClusterNumber, mode='distance', include_self=True).toarray()
    weight_matrix_m = np.square(weight_matrix_m)
    
    weight_matrix = b_i * weight_matrix_w + [1-b_i] * weight_matrix_m
    
    for i in range(mslic.ClusterNumber):
        test = weight_matrix[i,:]
        s_test = np.sort(test)        
        test[np.where(test > s_test[neighbours])] = 0
    
    for i in range(mslic.ClusterNumber):
        for j in range(i+1,mslic.ClusterNumber):
            if(weight_matrix[i,j] != weight_matrix[j,i]):
                weight_matrix[i,j] = weight_matrix[j,i] = np.maximum(weight_matrix[i,j],weight_matrix[j,i])
            
    weight_matrix = np.exp(-weight_matrix/(sigma_i*sigma_i))
    weight_matrix[np.where(weight_matrix ==1)] = 0
    
    return weight_matrix

def lgc_classifier(mark_2,node_labels,mu_i,mode):
    graph_solver = graph_semi_supervised(mark_2,node_labels)
    lgc_probs = graph_solver.lgc_solver(mu = mu_i)   
    if(mode == 0):
     for i in range(lgc_probs.shape[0]):
         lgc_probs[i,:] = lgc_probs[i,:] / np.sum(lgc_probs[i,:])
    if(mode == 1):
        lgc_probs = np.argmax(lgc_probs,axis =1) + 1 
    return lgc_probs


def ls_classifier(weight_matrix,test_labels,alpha):
    graph_solver = graph_semi_supervised(weight_matrix,test_labels)
    ls_labels = graph_solver.LS_solver(alpha = alpha)
    ls_labels = np.argmax(ls_labels,axis =1) + 1 
    return ls_labels


def graph_clustering(weight_matrix,test_labels,mu):
    graph_solver = graph_semi_supervised(weight_matrix,test_labels)    
    d_neg_half = np.diag(np.power(graph_solver.degrees,-0.5))
    S = np.linalg.multi_dot((d_neg_half,weight_matrix,d_neg_half))
    lag = np.identity(graph_solver.node_number) - S
    eigenvectors, eigenvalues, vh = np.linalg.svd(lag)

    eigenvalues = eigenvalues[::-1]
    eigenvectors = np.flip(eigenvectors,axis = 1)
    
    u_0 = np.copy(graph_solver.Y)
    fid = graph_solver.F
    u = np.copy(u_0)

    dt = 0.01     
    d = np.zeros((eigenvectors.shape[1],u_0.shape[1]))
    
    for i in range(20):        
        a = np.dot(np.transpose(eigenvectors),u)
        for i in range(a.shape[0]):
            a[i,:] = (1-dt*eigenvalues[i])*a[i,:] - dt*d[i,:]
        u = np.dot(eigenvectors,a)
        lhs = u - u_0
        for k in range(lhs.shape[1]):
            lhs[:,k] = mu*(fid[k]*lhs[:,k])
        d = np.dot(np.transpose(eigenvectors),lhs)
        
        u[np.arange(len(u)), u.argmax(1)] = 1
        u[np.where( u != 1)] = 0   
        
    ## Producing Layout
    produced_labels = np.zeros(u.shape[0],int)
    for j in range(u.shape[0]):
        produced_labels[j] = np.argmax(u[j,:])+1
    
    return produced_labels




def fcf(hms,gamma,sigma,b,mu,k,node_labels,mean_feature_vector,weighted_feature_vector):    

    cluster_positions = np.zeros((hms.ClusterNumber,2))
    cluster_count = np.zeros((hms.ClusterNumber))
    for i in range(hms.Height):
        for j in range(hms.Width):
            cluster_count[hms.Labels[i,j]] += 1 
            cluster_positions[hms.Labels[i,j],0] += i 
            cluster_positions[hms.Labels[i,j],1] += j
    
    
    for i in range(hms.ClusterNumber):
        cluster_positions[i,:] = cluster_positions[i,:] / cluster_count[i]   

    
    weight_matrix = mean_weighted_k_graph(hms,weighted_feature_vector,mean_feature_vector,sigma,b,hms.ClusterNumber-1) 
    distances = np.asarray(graph.create_distance_matrix(cluster_positions))       
    distances_matrix = np.exp(-distances/gamma)   
    
    mark_2 = np.multiply(weight_matrix,distances_matrix)

    neighbours = 8
    for i in range(hms.ClusterNumber):
        test = mark_2[i,:]
        s_test = np.sort(test)        
        test[np.where(test < s_test[-neighbours])] = 0
    
    for i in range(hms.ClusterNumber):
        for j in range(i+1,hms.ClusterNumber):
            if(mark_2[i,j] != mark_2[j,i]):
                mark_2[i,j] = mark_2[j,i] = np.maximum(mark_2[i,j],mark_2[j,i])

    v_2_labels= lgc_classifier(mark_2,node_labels,mu,1)    

    
    return v_2_labels


def lgc_classifier_m(mark_2,node_labels,mu_i,mode):
    graph_solver = graph_semi_supervised(mark_2,node_labels)
    lgc_probs = graph_solver.lgc_solver(mu = mu_i)   
    if(mode == 0):
     for i in range(lgc_probs.shape[0]):
         lgc_probs[i,:] = lgc_probs[i,:] / np.sum(lgc_probs[i,:])
    if(mode == 1):
        lgc_probs = np.argmax(lgc_probs,axis =1) + 1 
    return lgc_probs


def weighted_graph_m(feature_vector,sigma_i,b_i,neighbours):
    ## Create the weight matrix for the weighted features
    weight_matrix = kneighbors_graph(feature_vector, feature_vector.shape[0], mode='distance', include_self=True).toarray()
    weight_matrix = np.square(weight_matrix)    
    weight_matrix = np.exp(-weight_matrix/(sigma_i*sigma_i))
    np.fill_diagonal(weight_matrix,0)    
    return weight_matrix



def fcf_m(sigma,b,mu,k,node_labels,feature_vector):        
    weight_matrix = weighted_graph_m(feature_vector,sigma,b,feature_vector.shape[0]-1)     
    neighbours = 8
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
    print(final-start)
    # %%   
    return v_2_labels