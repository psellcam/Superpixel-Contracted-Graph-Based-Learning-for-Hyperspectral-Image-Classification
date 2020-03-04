#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 09:17:18 2018

@author: ps644
"""
import numpy as np
import math
from scipy import ndimage as ndi
import HMSCython

# The HMS Processor 
class HMSProcessor(object):    
        
    # Initialisation function
    def __init__(self,image,lcmr_m,k,m,a_1,a_2):         
        
        # Reads in the image
        self.ImageReading(image)
        # Dimensions of the image
        self.Height, self.Width, self.Bands  = self.ImageArray.shape[:3]        
        self.PixelNumber = self.Height * self.Width    
        
        # Variables which relate more to the processes rather than the image
        self.Areas = np.zeros((self.Height,self.Width))
        self.OriginalClusterNumber = k
        self.ClusterNumber = k
        self.Compactness = m        
        self.GridStep = int(math.sqrt(self.PixelNumber/self.ClusterNumber))
        self.iter = 0 
        self.residual_error = float("inf")
        self.ratio = 0
        self.splitting_threshold = 0.5  
        self.lcmr = lcmr_m
        
        ## Whilst not used in the paper you can weight the two spectral distances. 
        self.a_1 = a_1
        self.a_2 = a_2
        
        # Arrays to store the labels and distances used in assignemnt and update steps
        self.Labels = -1*np.ones((self.Height, self.Width),dtype=int)        
        self.Distances = np.full((self.Height, self.Width), 10000.0)    


    # Reads in the image and applies a small amount of Gaussian bluring which improves performance
    def ImageReading(self, image):     
        self.ImageArray = ndi.gaussian_filter(image , 0.1)            


    # Function to initialise the cluster number and locations
    def init_cluster(self):  
        countClusters = 0              
        i = int(self.GridStep/2)
        j = int(self.GridStep/2)
        
        number_of_clusters = (int((self.Height -  int(self.GridStep/2))/(self.GridStep) + 1))*(int((self.Width -  int(self.GridStep/2))/(self.GridStep) + 1))
        self.Clusters = np.zeros((number_of_clusters,4+self.Bands))
        self.ClustersLCMR = np.zeros((self.lcmr.shape[0],self.lcmr.shape[1],number_of_clusters))
        
        while( i < self.Height):
            j = int(self.GridStep/2)            
            while(j < self.Width):           
                self.Clusters[countClusters,0:2] = [i,j]
                self.Clusters[countClusters,2:(2+self.Bands)] = self.ImageArray[i,j,:]
                self.ClustersLCMR[:,:,countClusters] = self.lcmr[:,:,i,j]
                countClusters += 1
                j += self.GridStep
            i += self.GridStep   
            
        self.ClusterNumber = countClusters 


    # Function that gets the gradient for a pixel in the array     
    def get_gradient(self,h,w):
        if h + 1 >= self.Height:
            h = self.Height - 2
        if w + 1 >= self.Width:
            w = self.Width -2 
                
        verticalVector = self.ImageArray[h+1,w,:] - self.ImageArray[h-1,w,:]
        horizontalVector = self.ImageArray[h,w+1,:] - self.ImageArray[h,w-1,:]
        
        gradient = np.sqrt(np.dot(verticalVector,verticalVector) + np.dot(horizontalVector,horizontalVector))
        return gradient
    
    # Using the gradient function to check the starting positions of clusters and moves to a 
    # lower gradient position which is less likely to be an edge etc
    def gradient_search(self):
        for i  in range(self.ClusterNumber):
            cluster_gradient = self.get_gradient(int(self.Clusters[i,0]),int(self.Clusters[i,1]))
            minimum_gradient = cluster_gradient
            minimum_h = self.Clusters[i,0]
            minimum_w = self.Clusters[i,1]
            for dh in range(-1,2):
                _h = self.Clusters[i,0] + dh
                for dw in range(-1,2):                    
                    _w = self.Clusters[i,1] + dw
                    new_gradient = self.get_gradient(int(_h),int(_w))
                    if(new_gradient < minimum_gradient):
                       minimum_gradient = new_gradient
                       minimum_h = _h
                       minimum_w = _w                          
            self.Clusters[i,0] = int(minimum_h)       
            self.Clusters[i,1] = int(minimum_w)
            self.Clusters[i,2:2+self.Bands] = self.ImageArray[int(minimum_h),int(minimum_w),:]
            self.ClustersLCMR[:,:,i] = self.lcmr[:,:,int(minimum_h),int(minimum_w)]


    # Creates the manifold mapping of the pixels into R^2
    def manifold_area_creation(self):           
        self.Areas = np.asarray(HMSCython.area_element(self.ImageArray,self.Compactness,self.GridStep))       
        self.total_area = np.sum(self.Areas)
        self.local_search_range = (4*self.total_area)/self.ClusterNumber      
        


    # Calls find_area_search_limit for each seed to find the adaptive search regions
    def scaling_calculation(self):
        for c_i in range(self.ClusterNumber):  
            ch = int(self.Clusters[c_i,0])
            cw = int(self.Clusters[c_i,1])  
            area_sum = HMSCython.find_area_search_limit(ch,cw,self.GridStep,self.Areas)
            self.Clusters[c_i,3+self.Bands] = np.sqrt(self.local_search_range/area_sum)
            
    # Split seeds which match certain conditions
    def splitting_cells(self):           
        split_count = 0
        # Stores whether a seed should be split
        split_status = np.zeros(self.ClusterNumber)
        for c_i in range(self.ClusterNumber):             
            # Conditions which monitor if things should be split              
            if(self.iter > 0 and self.Clusters[c_i,3+self.Bands] < self.splitting_threshold and self.Clusters[c_i,2+self.Bands] > self.local_search_range/4):
                split_count += 1
                split_status[c_i] = 1   
                
        # Store new clusters , each split creates three more
        self.new_clusters = np.zeros((self.ClusterNumber+3*split_count,4+self.Bands))
        self.new_lcmr = np.zeros((self.lcmr.shape[0],self.lcmr.shape[1],self.ClusterNumber+3*split_count))
        
        # Monitors the transformation for the labels due to changing clusters
        self.split_transformation = np.zeros((self.ClusterNumber))
        new_cluster_index = 0;
        
        for c_i in range(self.ClusterNumber):
            if(split_status[c_i] == 0):
                # If it doesnt need to split just copy it across to the current index and add one
                self.new_clusters[new_cluster_index,:] = self.Clusters[c_i,:]
                self.new_lcmr[:,:,new_cluster_index] = self.ClustersLCMR[:,:,c_i]
                self.split_transformation[c_i] = new_cluster_index
                new_cluster_index += 1
                
                
            if(split_status[c_i] == 1):
                
                # Need this to add the 4 new cluster positions
                cluster_h = int(self.Clusters[c_i,0])
                cluster_w = int(self.Clusters[c_i,1])
                displacement = int(round((self.GridStep*self.Clusters[c_i,3+self.Bands])/2))                
               
                min_h = cluster_h - displacement
                max_h = cluster_h + displacement
                min_w = cluster_w - displacement
                max_w = cluster_w + displacement
                
                # Checks that the new places aren't outside the actual image
                if(min_h < 0):
                    min_h = 0 
                if(min_w < 0):
                    min_w = 0
                if(max_h > self.Height - 1):
                    max_h = self.Height - 1
                if(max_w > self.Width - 1):
                    max_w = self.Width - 1
                
                
                # Adds the 4 new clusters to the new_clusters array 
                self.new_clusters[new_cluster_index,0] = min_h
                self.new_clusters[new_cluster_index,1] = max_w
                self.new_clusters[new_cluster_index,2:(2+self.Bands)] = self.ImageArray[min_h,max_w,:]
                self.new_clusters[new_cluster_index,2+self.Bands] = self.local_search_range
                self.new_clusters[new_cluster_index,3+self.Bands] = self.Clusters[c_i,3+self.Bands]/2
                self.new_lcmr[:,:,new_cluster_index] = self.lcmr[:,:,min_h,max_w]
                
                
                self.new_clusters[new_cluster_index+1,0] = min_h
                self.new_clusters[new_cluster_index+1,1] = min_w
                self.new_clusters[new_cluster_index+1,2:(2+self.Bands)] = self.ImageArray[min_h,min_w,:]
                self.new_clusters[new_cluster_index+1,2+self.Bands] = self.local_search_range
                self.new_clusters[new_cluster_index+1,3+self.Bands] = self.Clusters[c_i,3+self.Bands]/2
                self.new_lcmr[:,:,new_cluster_index+1] = self.lcmr[:,:,min_h,min_w]
                
                self.new_clusters[new_cluster_index+2,0] = max_h
                self.new_clusters[new_cluster_index+2,1] = max_w
                self.new_clusters[new_cluster_index+2,2:(2+self.Bands)] = self.ImageArray[max_h,max_w,:]
                self.new_clusters[new_cluster_index+2,2+self.Bands] = self.local_search_range
                self.new_clusters[new_cluster_index+2,3+self.Bands] = self.Clusters[c_i,3+self.Bands]/2
                self.new_lcmr[:,:,new_cluster_index+2] = self.lcmr[:,:,max_h,max_w]
                
                self.new_clusters[new_cluster_index+3,0] = max_h
                self.new_clusters[new_cluster_index+3,1] = min_w
                self.new_clusters[new_cluster_index+3,2:(2+self.Bands)] = self.ImageArray[max_h,min_w,:]
                self.new_clusters[new_cluster_index+3,2+self.Bands] = self.local_search_range
                self.new_clusters[new_cluster_index+3,3+self.Bands] = self.Clusters[c_i,3+self.Bands]/2
                self.new_lcmr[:,:,new_cluster_index+3] = self.lcmr[:,:,max_h,min_w]
                
                #Only need to map the labels for the first new clusters as its labels are the original clusters before splitting
                self.split_transformation[c_i] = new_cluster_index
                new_cluster_index += 4             
                
                
        self.Clusters = self.new_clusters
        self.ClusterNumber = self.ClusterNumber + 3*split_count
        self.ClustersLCMR = self.new_lcmr
        
        for i in range(self.Height):
            for j in range(self.Width):
                lbl = self.Labels[i,j]
                self.Labels[i,j] = self.split_transformation[int(lbl)]

    # Merge two clusters together
    def merging(self,c_i,label):
        # Adds the cluster to the visited ones
        self.visited_status[label] = 1
        # Find where it is        
        label_positions = self.positions[label,:self.counts[label],:]
        # Change the labels
        for i in range(label_positions.shape[0]):
            self.Labels[label_positions[i,0],label_positions[i,1]] = c_i

        
        # Create the new cluster instead of the old one as per the paper
        new_cluster = self.Clusters[c_i,:]*self.Clusters[c_i,(2+self.Bands)] + self.Clusters[label,:]*self.Clusters[label,(2+self.Bands)]
        new_cluster = new_cluster / (self.Clusters[c_i,(2+self.Bands)] + self.Clusters[label,(2+self.Bands)])
        new_cluster[0] = round(new_cluster[0])
        new_cluster[1] = round(new_cluster[1])
        new_cluster[(2+self.Bands)] = self.Clusters[c_i,(2+self.Bands)] + self.Clusters[label,(2+self.Bands)]
        self.Clusters[c_i,:] = new_cluster
        self.ClustersLCMR[:,:,c_i] = 0.5*self.ClustersLCMR[:,:,c_i] + 0.5*self.ClustersLCMR[:,:,label]
        # Set the cluster which has been consumed to zero
        self.Clusters[label,:] = np.zeros((4+self.Bands))
        self.ClustersLCMR[:,:,label] = np.zeros((self.ClustersLCMR.shape[0],self.ClustersLCMR.shape[0]))
    
    # Merges cell that are in content sparse areas
    def merging_cells(self):           
        # Visited status of all cluster set to -1
        self.visited_status = -1 * np.ones((self.ClusterNumber),dtype = int)        
        [self.positions,self.counts] = HMSCython.where_cluster(self.Labels,self.ClusterNumber)
        merging_count = 0
        for c_i in range(self.ClusterNumber):
            #If it hasnt been visited
            if(self.visited_status[c_i] == -1):
                # Put it as visited , find univisited neighbouring clusters
                self.visited_status[c_i] = 1
                c_i_positions = self.positions[c_i,:self.counts[c_i],:]  
                [neighbours,count] = HMSCython.neighbour_search(c_i_positions,self.Labels,self.visited_status,self.GridStep)
                neighbours = list(set(neighbours[:count]))
                
                # One possible merge
                if(neighbours and self.Clusters[c_i,(2+self.Bands)] < self.local_search_range/32):                          
                    neighbour_differences = np.zeros((len(neighbours)))
                    for i in range(len(neighbours)):
                        ## Merge it with the most similar looking cluster
                        neighbour_differences[i] = np.linalg.norm(self.Clusters[c_i,2:(2+self.Bands)] - self.Clusters[neighbours[i],2:(2+self.Bands)])
                    chosen_label = neighbours[np.argmin(neighbour_differences)]
                    self.merging(c_i,chosen_label)
                    merging_count += 1
                    
                    
                elif(neighbours and self.Clusters[c_i,(2+self.Bands)] < self.local_search_range/5):
                    ## Merge it with the neighbouring cluster which makes the smallest area
                    combined_areas = np.zeros(len(neighbours))
                    for i in range(len(neighbours)):
                        combined_areas[i] = (self.Clusters[c_i,(2+self.Bands)] + self.Clusters[neighbours[i],(2+self.Bands)])/(self.local_search_range/5)
                    if(np.amin(combined_areas) < 1):
                        chosen = np.argmin(combined_areas)
                        chosen_label = neighbours[chosen]
                        self.merging(c_i,chosen_label)
                        merging_count += 1


        #Find the transformation of the clusters removing the ones that has been removed
        self.merge_transformation =  np.zeros(self.ClusterNumber)
        new_clusters = np.zeros((self.ClusterNumber- merging_count,(4+self.Bands)))
        new_lcmr = np.zeros((self.ClustersLCMR.shape[0],self.ClustersLCMR.shape[0],self.ClusterNumber- merging_count))
        
        placing_index = 0 
        for c_i in range(self.ClusterNumber):
            if(self.Clusters[c_i,3] != 0):
                new_clusters[placing_index,:] = self.Clusters[c_i,:]       
                new_lcmr[:,:,placing_index] = self.ClustersLCMR[:,:,c_i]
                self.merge_transformation[c_i] = placing_index
                placing_index += 1 


        # Update the cluster number and the clusters
        self.ClusterNumber = self.ClusterNumber - merging_count
        #print("Merging count", merging_count, "New cluster number",self.ClusterNumber)
        self.Clusters = new_clusters   
        self.ClustersLCMR = new_lcmr
                
                
    # Calls the cython code to assign labels to pixels from the current cluster conditions                          
    def assignmentCython(self):       
        
        # Creates array structures for the clusters for use in the Cython code
        clusterColours = np.zeros((self.ClusterNumber,self.Bands))
        clusterPositions = np.zeros((self.ClusterNumber,2),dtype=int)
        clusterScales = np.ascontiguousarray(self.Clusters[:,3+self.Bands])
        clusterPositions = np.ascontiguousarray(self.Clusters[:,0:2],dtype=int)
        clusterColours = np.ascontiguousarray(self.Clusters[:,2:(2+self.Bands)])
        clusterLCMR = np.ascontiguousarray(self.ClustersLCMR)
        
        # Resets the distance array
        distanceArray = np.full((self.Height, self.Width), 10000.0)        
        
        # Creating variables for the function. Not really necessary
        imageGrid = np.copy(self.ImageArray)             
        cluster_number = int(self.ClusterNumber)
        gridStep = int(self.GridStep)+1
        compactness = float(self.Compactness)      
         
        [self.Labels,self.Distances] = np.asarray(HMSCython.assign_clusters_cython(imageGrid,gridStep,cluster_number,distanceArray,clusterPositions,clusterColours,clusterLCMR,clusterScales,compactness,self.lcmr,self.a_1,self.a_2))
        self.Labels = self.Labels.astype(int)      



    # Updates the clusters using the current labels.    
    def updateClustersCython(self):        
        # Creating variables for the function. Not really necessary
       
        old_residual = self.residual_error
        if(self.Clusters[0,0] > 0):
            self.residual_error = 0
            
            
            # The updated cluster information is stored in the kvector_info. In the form h,w,L,a,b
            [kvector_info,kvector_lcmr] = np.asarray(HMSCython.update_cluster_cython(self.ImageArray,self.lcmr,self.Labels,self.ClusterNumber,self.Areas))               
            # Updates the position colour and area but leaves the scaling factor the same
            for i in range(self.ClusterNumber):         
                if(kvector_info[0,i] != 10000.0):  
                    error = np.linalg.norm(self.Clusters[i,:(2+self.Bands)] - kvector_info[:(2+self.Bands),i])**2
                    self.residual_error += error                
                    self.Clusters[i,:(3+self.Bands)] = kvector_info[:,i]  
                    self.residual_error += np.linalg.norm(self.ClustersLCMR[:,:,i] - kvector_lcmr[:,:,i],'fro')
                    self.ClustersLCMR[:,:,i] = kvector_lcmr[:,:,i]
                
                # If for some reason a seed produces no labels randomly place it in the image
                if(kvector_info[0,i] == 10000.0):                                                    
                    self.Clusters[i,0] += round(self.GridStep/4) 
                    if(self.Clusters[i,0] > self.Height-1):
                        self.Clusters[i,0] = self.Height - 3
                    self.Clusters[i,1] += round(self.GridStep/4)
                    if(self.Clusters[i,1] > self.Width-1):
                        self.Clusters[i,1] = self.Width - 3
                    self.Clusters[i,2:(2+self.Bands)] = self.ImageArray[int(self.Clusters[i,0]),int(self.Clusters[i,1]),:]
                    self.Clusters[i,(2+self.Bands)] = self.local_search_range/4.5            
            
                    self.ClustersLCMR[:,:,i] = self.lcmr[:,:,int(self.Clusters[i,0]),int(self.Clusters[i,1])]   
                    
                    
            if(self.residual_error != 0):
                self.ratio = self.residual_error / old_residual       
            else:
                self.ratio = 1

           
        # Prevent residual error being calculated for the possible update after connectivity            
        elif(self.Clusters[0,3] == 0):
            # The updated cluster information is stored in the kvector_info. In the form h,w,L,a,b
            [kvector_info,kvector_lcmr] = np.asarray(HMSCython.update_cluster_cython(self.ImageArray,self.lcmr,self.Labels,self.ClusterNumber,self.Areas))                       
            # Updates the position colour and area 
            for i in range(self.ClusterNumber):                                 
                    self.Clusters[i,:(3+self.Bands)] = kvector_info[:,i]     
                    self.ClustersLCMR[:,:,i] = kvector_lcmr[:,:,i]    
                    self.Clusters[i,0] = round(self.Clusters[i,0])
                    self.Clusters[i,1] = round(self.Clusters[i,1])
                    
                    
    # The connectivity between the clusters is enforced. Also the minimum cluster size is enforced as well.
    def enforceConnectivity(self):
        segment_size = self.Height * self.Width / self.OriginalClusterNumber
        ## Bit random atm
        ## Hyper Parameters
        min_size = int(8)
        max_size = int(10*segment_size)        
        segments = np.copy(self.Labels,)  
        segments = segments.astype(int)
           
        self.Labels = HMSCython.connectivity_cython(segments,min_size,max_size)                  
        self.ClusterNumber = np.amax(self.Labels)+1
        self.Clusters = np.zeros((self.ClusterNumber,(4+self.Bands)))
        self.ClustersLCMR = np.zeros((self.ClustersLCMR.shape[0],self.ClustersLCMR.shape[0],self.ClusterNumber))
        self.updateClustersCython()
            #print("Final cluster number is", self.ClusterNumber)
    
           
    # Main work flow. This is the important function
    def main_work_flow(self):
        # Sets the initial cluster positions 
        self.init_cluster()
        # We want the superpixel representation to be a contraction so rejection if too many superpixels.
        if(self.PixelNumber / self.ClusterNumber < 15):
            raise ValueError('Too many clusters have been asked for. Please try a smaller value for k.')
        # Move superpixels into low gradient positions
        self.gradient_search()
        # Calculate manifold area
        self.manifold_area_creation()
        
        
        # Convergence condition
        while(self.ratio < 0.99 or self.ratio > 1 and self.iter < 2):                 
            #Main loop.            
            self.scaling_calculation()

            if(self.iter > 0):                
                self.splitting_cells()                 
                self.merging_cells()
                
            
            self.assignmentCython()               
            self.updateClustersCython()   
            self.iter += 1              
        self.enforceConnectivity()       
        return self.Labels
   