#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 13:29:18 2018

@author: ps644
"""

import numpy as np
import math
from sklearn.neighbors import kneighbors_graph
from sklearn.semi_supervised import LabelPropagation,LabelSpreading


def mean_spectral_feature(spectral_original,mslic):    
    cluster_labels = mslic.Labels
    mean_feature_vector = np.zeros((mslic.ClusterNumber,spectral_original.shape[2]))
    super_pixel_count = np.zeros((mslic.ClusterNumber))
    
    for i in range(spectral_original.shape[0]):
        for j in range(spectral_original.shape[1]):
                super_pixel_count[cluster_labels[i,j]] += 1 
                mean_feature_vector[cluster_labels[i,j],:] += spectral_original[i,j,:]


    for i in range(mslic.ClusterNumber):
        mean_feature_vector[i,:] = mean_feature_vector[i,:] / super_pixel_count[i]     

    return mean_feature_vector 



def weighted_spectral_feature(mean_feature_vector,hms,h):
    cluster_labels = hms.Labels
    adjacency_matrix = np.zeros((hms.ClusterNumber,hms.ClusterNumber))
    weighted_feature_vector = np.zeros((hms.ClusterNumber,mean_feature_vector.shape[1]))
    
    di = [1,-1,0,0]
    dj = [0,0,1,-1]
    for i in range(1,cluster_labels.shape[0]-1):
        for j in range(1,cluster_labels.shape[1]-1):
            label = hms.Labels[i,j]
            for s in range(4):
                s_i = i + di[s]
                s_j = j + dj[s]
                if(label != hms.Labels[s_i,s_j]):
                    adjacency_matrix[label,hms.Labels[s_i,s_j]] = 1 
    
    
#    for i in range(hms.ClusterNumber):
#        for j in range(i+1,hms.ClusterNumber):
#            if(np.sqrt(np.square(hms.Clusters[i,0] - hms.Clusters[j,0]) + np.square(hms.Clusters[i,1] - hms.Clusters[j,1])) < 3*hms.GridStep):
#                adjacency_matrix[i,j] = adjacency_matrix[j,i] =  1 
            
    neigh_num = np.zeros(hms.ClusterNumber,dtype=int)
    for i in range(hms.ClusterNumber):
        neighbours = np.where(adjacency_matrix[i,:] == 1)[0]
        neigh_num[i] = len(neighbours)   
        weights  = np.zeros((neigh_num[i]))
        for w_i in range(len(neighbours)):
            dist = [(a - b)**2 for a, b in zip(mean_feature_vector[i,:], mean_feature_vector[neighbours[w_i],:])]
            dist = sum(dist)
            weights[w_i] = dist
            weights[w_i] = math.exp(-dist/h)                  
                          
        if(sum(weights) > 0):
            weights = weights / sum(weights)
        
            
        if(sum(weights) > 0):            
            for w_i in range(len(neighbours)):
                weighted_feature_vector[i] += weights[w_i]*mean_feature_vector[neighbours[w_i],:]
        else:
            ## If the weight to its neighbours is so weak just consider its own neighbourhood 
            ## TODO is this the best way
            print('Node was too different to surrounding neighbours so weighted vector is mean vector.')
            weighted_feature_vector[i] = mean_feature_vector[i]
            
    return  weighted_feature_vector
        
    

def label_spreading(spectral_original,sparse_ground_truth,mslic,cluster_number):
    labels = np.copy(mslic.Labels)
    indexs = np.where(labels == cluster_number)
    
    cluster_spectral = spectral_original[indexs]
    cluster_labels = sparse_ground_truth[indexs]
    cluster_labels = cluster_labels.astype(int)
    
    
    cluster_labels[np.where(cluster_labels == 0)] = -1 
    current_value = 0 
    transformation = []
    for i in range(cluster_labels.shape[0]):
        if(cluster_labels[i] != -1 and cluster_labels[i] > current_value):         
            value = cluster_labels[i]          
            transformation.append([value,current_value])
            cluster_labels[np.where(cluster_labels == value)] = current_value
            current_value += 1 
    
    
    label_prop_model = LabelSpreading(kernel='rbf',gamma = (1/1000000),n_jobs = -1)    
    label_prop_model.fit(cluster_spectral, cluster_labels)
    output_labels = label_prop_model.transduction_    
    output_labels[np.where(output_labels == 0)] = cluster_number
    
    current_value = mslic.ClusterNumber
    for i in range(output_labels.shape[0]):
        if(output_labels[i] < len(transformation)):         
            value = output_labels[i]                    
            output_labels[np.where(output_labels == value)] = current_value
            current_value += 1 
            
            
    for i in range(output_labels.shape[0]):
        p_i = indexs[0][i]
        p_j = indexs[1][i]

        labels[p_i,p_j] = output_labels[i]
        
    mslic.Labels = labels
    mslic.ClusterNumber = np.amax(mslic.Labels)+1