i#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 14:50:51 2019

@author: ps644
"""

# -*- coding: utf-8 -*-
import time
import numpy as np 
import data_analysis
import lcmr_functions as lcmr
import processing_data as pd
import classification_functions as cf
from sklearn.preprocessing import MinMaxScaler
import sklearn.metrics
import HMS
import matplotlib.pyplot as plt
##################################################################################################################
#### Start of the main code

#### Load Hyperspectral Data and Ground Truth
time_1 = time.time()
## Different names are Indian, Salinas and PaviaU
name = 'Indian'
[spectral_original,ground_truth] = data_analysis.loading_hyperspectral(name)
no_class = np.max(ground_truth)
time_2 = time.time()
print("Time to load data", time_2 - time_1)

#### Perform dimesnionality reduction using PCA. Extracts enough components to meet required variance.
time_1 = time.time() 
spectral_data_pca = data_analysis.principal_component_extraction(spectral_original,0.998)
time_2 = time.time()
print("Time to perform PCA", time_2 - time_1)

#### LCMR Matrix Construction
print("Time to construct covariance matrices", end =" ")
start = time.time()
spectral_mnf = lcmr.dimensional_reduction_mnf(spectral_original,20)
lcmr_matrices = lcmr.create_logm_matrices(spectral_mnf,25,400)
final = time.time()
print("Completion time", final-start)

##### HMS Over-segmentation
print("Time to over-segment using HMS", end =" ")
start = time.time()
#### k Indian Pines 1200 Salinas 1500 Pavia Uni 2400
hms = HMS.HMSProcessor(image = spectral_data_pca,lcmr_m = lcmr_matrices, k=400, m = 6, a_1 = 0.5, a_2 = 0.5)
labels = hms.main_work_flow()
final = time.time()
print("Completion time", final-start)


imgColor = spectral_original[:,:,[19 ,15 ,12]]
imgColor = imgColor.astype(np.double)
imgColor = 255*(imgColor - np.amin(imgColor)) / (np.amax(imgColor)-np.amin(imgColor))
imgColor = imgColor.astype(np.uint8)
cluster_averages = np.zeros((hms.ClusterNumber,3))
cluster_counts = np.zeros((hms.ClusterNumber))

for i in range(labels.shape[0]):
    for j in range(labels.shape[1]):
        cluster_averages[labels[i,j],:] += imgColor[i,j,:]
        cluster_counts[labels[i,j]] += 1

for i in range(hms.ClusterNumber):
    cluster_averages[i,:] = cluster_averages[i,:] / cluster_counts[i]

for i in range(labels.shape[0]):
    for j in range(labels.shape[1]):
        imgColor[i,j,:] = cluster_averages[labels[i,j],:]
        
# Saves the image and the superpixel representation
mean_image = np.copy(imgColor)
plt.imshow(mean_image)
plt.imsave("mean.png",mean_image)  

imgColor2 = spectral_original[:,:,[29 ,15 ,12]]
imgColor2 = 255*(imgColor2 - np.amin(imgColor2)) / (np.amax(imgColor2)-np.amin(imgColor2))
imgColor2 = imgColor2.astype(np.uint8)

test = hms.Clusters[:,:2]

for i in range(imgColor2.shape[0]):
    for j in range(imgColor2.shape[1]):
        
        for s in range(4):
            di = np.asarray([1,0,0,-1])
            dj = np.asarray([0,1,-1,0])
            
            s_i = i + di[s]
            s_j = j + dj[s]
            
            if(s_i < imgColor.shape[0] and s_j < imgColor.shape[1]):
                if(labels[s_i,s_j] != labels[i,j]):
                    imgColor2[i,j,:] = [255,0,0]
     
boundary_image = np.copy(imgColor2)
plt.imshow(boundary_image)
plt.imsave("boundary.png",boundary_image)  