# -*- coding: utf-8 -*-
# This code was written by Philip Sellars if you use this code in your research
# please cite the associated paper. For any questions or queries please email me at 
# ps644@cam.ac.uk


# %% Modules to Import
import time
import numpy as np 
import data_analysis
import lcmr_functions as lcmr
import processing_data as pd
import classification_functions as cf
from sklearn.preprocessing import MinMaxScaler
import sklearn.metrics
import sklearn.manifold
import random
import HMS
import graphclass
import lcmr_cython
import matplotlib.pyplot as plt
import cli
 

args = cli.parse_commandline_args()
if args.dataset == "Indian":
    args.k = 1200
    args.b = random.uniform(0.89,0.9) 
    args.gamma = 10**random.uniform(-0.8,-0.6)  
elif args.dataset == "Salinas":
    args.k = 1500
    args.b = random.uniform(0.89,0.9) 
    args.gamma = 10**random.uniform(0.8,1.3)  
elif args.dataset == "PaviaU":
    args.k=2400
    args.b = random.uniform(0.09,0.1) 
    args.gamma = 10**random.uniform(2.5,3.0)  
    

#### Load Hyperspectral Data and Ground Truth
time_1 = time.time()
## Different names are Indian, Salinas and PaviaU
dataset = args.dataset
print("Using the",dataset,"dataset")
[spectral_original,ground_truth] = data_analysis.loading_hyperspectral(dataset)
no_class = np.max(ground_truth)
time_2 = time.time()
print("Time to load data", time_2 - time_1)
 
### Perform dimesnionality reduction using PCA. Extracts enough components to meet required variance.
time_1 = time.time() 
spectral_data_pca = data_analysis.principal_component_extraction(spectral_original,args.pca_v)
time_2 = time.time()
print("Time to perform PCA", time_2 - time_1)

### LCMR Matrix Construction
start = time.time()
spectral_mnf = lcmr.dimensional_reduction_mnf(spectral_original,args.cm_feats)
lcmr_matrices = lcmr.create_logm_matrices(spectral_mnf,args.cm_size,args.cm_nn)
final = time.time()
print("Time to produce covariance matrices",final-start)
      
##### HMS Over-segmentation
print("Time to over-segment using HMS", end =" ")
start = time.time()

hms = HMS.HMSProcessor(image = spectral_data_pca,lcmr_m = lcmr_matrices, k=args.k, m = 4,mc=True)
labels = hms.main_work_flow()
final = time.time()
print(final-start)

#### Cluster Ground Truth Information
[cluster_gt,cluster_contents] = data_analysis.cluster_arg_max_lable(ground_truth,hms)   
cluster_size = np.sum(cluster_contents,axis=1)
cluster_purity = np.amax(cluster_contents,axis=1)/np.sum(cluster_contents,axis=1)    

#### Number of runs to repeat with different selections of labels
#### Storage for the performance metrics
#### Overall accuracy, Kappa, class average
OA_store = np.zeros((args.run_n))
Kappa_store = np.zeros((args.run_n))
CA_store = np.zeros((no_class,args.run_n))
print("Feature Extraction and Graph Classification", end =" ")
start_1 = time.time()
pro_class_max = np.zeros(ground_truth.shape)
OA_max = 0
for t_i in range(args.run_n):      
    ### Extract sparse ground truth and the initial cluster labels
    sparse_ground_truth = pd.sparseness_operator(ground_truth,args.num_labeled,hms,spectral_original)
    [node_probs, node_labels] = pd.node_label_initialisation(sparse_ground_truth,hms)
    cluster_contents_sparse = pd.node_label_full(sparse_ground_truth,hms)      
  
    #### Get sparse mean and weighted feature vectors from the PCA reduced components
    mean_feature_vector_ori = cf.mean_spectral_feature(spectral_data_pca,hms)
    weighted_feature_vector_ori = cf.weighted_spectral_feature(mean_feature_vector_ori,hms,h=15)   
       
    #### Resize the feature to range [-1,1] 
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(mean_feature_vector_ori)    
    mean_feature_vector = scaler.transform(mean_feature_vector_ori)
    scaler.fit(weighted_feature_vector_ori)    
    weighted_feature_vector = scaler.transform(weighted_feature_vector_ori)    
    test_labels = np.copy(node_labels)
    test_data = np.copy(weighted_feature_vector)
          
    #### Parameter setting for the graph constrcution and classification
    sigma = random.uniform(0.19,0.20)
    mu = random.uniform(0.1,0.15)
    k = 8
 
  
    ## Graph Propogation 
    v_2_labels = graphclass.fcf(hms,args.gamma,sigma,args.b,mu,k,node_labels,mean_feature_vector,weighted_feature_vector)
    [final_accuracy_v_2,final_coverage,final_pixel_number] = pd.initial_accuracy_assessment(ground_truth,hms.Labels,v_2_labels)  
    pro_class = pd.produce_classification(v_2_labels,labels,ground_truth)
    final = time.time()    
  
    # I.E Remove the training set from the ground truth to not unfairly count them twice
    adjusted_ground_truth = np.copy(ground_truth)
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            if(sparse_ground_truth[i,j] > 0):
                adjusted_ground_truth[i,j] = 0
          
    ## Calculation Performance Metrices 
    pro_class_nb = np.copy(pro_class)
    pro_class_nb = pro_class_nb[np.where(adjusted_ground_truth  > 0)] 
    ground_truth_nb = adjusted_ground_truth[adjusted_ground_truth > 0]
  
    OA_store[t_i] = sklearn.metrics.accuracy_score(pro_class_nb, ground_truth_nb)
   
    if(OA_store[t_i] > OA_max):
        pro_class_max = np.copy(pro_class)
        OA_max = OA_store[t_i]
      
    Kappa_store[t_i] = sklearn.metrics.cohen_kappa_score(pro_class_nb, ground_truth_nb)
    test = sklearn.metrics.confusion_matrix(pro_class_nb, ground_truth_nb)
    test = test / np.sum(test,axis=0)
    CA_store[:,t_i] = np.diag(test)

print(time.time() - start_1)
print("Total time", time.time() - time_1)

## Store Performance Metrics for All Runs
performance_metrics = np.zeros((np.amax(ground_truth)+3,2))
performance_metrics[:no_class,0] = np.average(CA_store,axis=1)
performance_metrics[:no_class,1] = np.std(CA_store,axis=1)
performance_metrics[no_class,0] = np.average(OA_store)
performance_metrics[no_class,1] = np.std(OA_store)
performance_metrics[no_class+1,0] = np.average(performance_metrics[:no_class,0])
performance_metrics[no_class+1,1] = np.std(performance_metrics[:no_class,0])
performance_metrics[no_class+2,0] = np.average(Kappa_store)
performance_metrics[no_class+2,1] = np.std(Kappa_store)
performance_metrics = performance_metrics * 100

mean = "{:,.2f}".format(performance_metrics[no_class,0])
std = "{:,.2f}".format(performance_metrics[no_class,1])
print("Overall Accuracy",mean,"+-",std,"%")
    


## Extra Code for Saving images

#imgColor = spectral_original[:,:,[39 ,28 ,12]]
#imgColor = 255*(imgColor - np.amin(imgColor)) / (np.amax(imgColor)-np.amin(imgColor))
#imgColor = imgColor.astype(np.uint8)
#plt.imsave("pavia_colour.png",imgColor)

#from skimage.segmentation import mark_boundaries
#image_with_boundaries = mark_boundaries(imgColor,labels)
#plt.imsave('pavia_boundaries.png',image_with_boundaries)


# imgMean = np.zeros((imgColor.shape))
# for i in range(np.amax(labels)):
#     colour_group = imgColor[np.where(labels == i)]
#     imgMean[np.where(labels == i)] = np.average(colour_group,axis=0)
#     imgMean = imgMean.astype(int)
# plt.imsave("pavia_mean_2400.png",imgMean)