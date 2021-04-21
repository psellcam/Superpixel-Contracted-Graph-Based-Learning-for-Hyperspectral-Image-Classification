import scipy.io as sio
import os 
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
import time

### Function that loads the different hyperspectral datasets 
def loading_hyperspectral(name):
    ## Variable List
    ## Name: The given name of the hyperspectral dataset required
    
    ## Output List
    ## Spectral Data: The WxHxB hyperspectral image
    ## Ground Truth: The WxH ground truth labels   
    
    if(name == 'PaviaU'):
        current_path = os.getcwd()
        dir_path = current_path + "/PaviaUni"
    
        os.chdir(dir_path)    
        extracted_mat_lab_file = sio.loadmat('PaviaU.mat')     
        spectral_data = extracted_mat_lab_file['paviaU']    
        spectral_data = spectral_data.astype(float)
    
        extracted_mat_lab_file = sio.loadmat('PaviaU_gt.mat')   
        ground_truth = extracted_mat_lab_file['paviaU_gt']
    
        os.chdir(current_path)       
        return spectral_data , ground_truth
    
    if(name == 'Indiana'):
        
        # Go and get the spectral data and ground truth
        current_path = os.getcwd()
        dir_path = current_path + "/Indiana"
        
        os.chdir(dir_path)
    
        extracted_mat_lab_file = sio.loadmat('Indian_pines_corrected.mat')     
        spectral_data = extracted_mat_lab_file['indian_pines_corrected']    
        spectral_data = spectral_data.astype(float)
        
        extracted_mat_lab_file = sio.loadmat('Indian_pines_gt.mat')   
        ground_truth = extracted_mat_lab_file['indian_pines_gt']                         
    
        os.chdir(current_path)                
        return spectral_data , ground_truth
        
    if(name == 'Salinas'):
        current_path = os.getcwd()
        dir_path = current_path + "/Salinas"
        
        os.chdir(dir_path)
    
        extracted_mat_lab_file = sio.loadmat('Salinas_corrected.mat')     
        spectral_data = extracted_mat_lab_file['salinas_corrected']    
        spectral_data = spectral_data.astype(float)
        
        extracted_mat_lab_file = sio.loadmat('Salinas_gt.mat')   
        ground_truth = extracted_mat_lab_file['salinas_gt']    
    
        os.chdir(current_path)               
        return spectral_data , ground_truth
    
    if(name == 'Botswana'):
        current_path = os.getcwd()
        dir_path = current_path + "/Botswana"
        
        os.chdir(dir_path)
    
        extracted_mat_lab_file = sio.loadmat('Botswana.mat')     
        spectral_data = extracted_mat_lab_file['Botswana']    
        spectral_data = spectral_data.astype(float)
        
        extracted_mat_lab_file = sio.loadmat('Botswana_gt.mat')   
        ground_truth = extracted_mat_lab_file['Botswana_gt']    
    
        os.chdir(current_path)               
        return spectral_data , ground_truth
    
    if(name == 'KSC'):
        current_path = os.getcwd()
        dir_path = current_path + "/KSC"
        
        os.chdir(dir_path)
    
        extracted_mat_lab_file = sio.loadmat('KSC.mat')     
        spectral_data = extracted_mat_lab_file['KSC']    
        spectral_data = spectral_data.astype(float)
        
        extracted_mat_lab_file = sio.loadmat('KSC_gt.mat')   
        ground_truth = extracted_mat_lab_file['KSC_gt']    
    
        os.chdir(current_path)               
        return spectral_data , ground_truth
    
### Dimensionality reduction via PCA. The function returns the number of principal components needed to 
### reach the required variance ratio
        
def principal_component_extraction(spectral_original,variance_required):    
    ## Variable List
    ## spectral_original: The original non reduced image
    ## variance_required: The required variance  ratio from 0 to 1   
    
    ## Output list
    ## spectral_pc_final: The dimensional reduces image
    
    # 2d reshape
    spectral_2d = spectral_original.reshape((spectral_original.shape[0]*spectral_original.shape[1],spectral_original.shape[2]))
    # Feature scaling preprocessing step
    spectral_2d =  preprocessing.scale(spectral_2d)
 
    if(spectral_2d.shape[1] < 100):
        pca  = PCA(n_components=spectral_2d.shape[1])
    else:
        pca  = PCA(n_components=100)
    spectral_pc = pca.fit_transform(spectral_2d)
    explained_variance = pca.explained_variance_ratio_
    
    if(np.sum(explained_variance) < variance_required):
        raise ValueError("The required variance was too high. Values should be between 0 and 1.")
    
    # Select the number of principal components that gives the variance required
    explained_variance_sum = np.zeros(explained_variance.shape)
    sum_ev = 0
    component_number = 0 
    for i in range(explained_variance.shape[0]):
        sum_ev += explained_variance[i]
        if (sum_ev > variance_required and component_number == 0):
            component_number = i+1
        explained_variance_sum[i] = sum_ev
        
    
        
    # Removed the unnecessary components and reshape in original 3d form 
    spectral_pc = spectral_pc[:,:component_number]
    spectral_pc_final = spectral_pc.reshape((spectral_original.shape[0],spectral_original.shape[1],component_number))

    return spectral_pc_final


### Function that returns the most common class in a superpixel and the class by class breakdown
def cluster_arg_max_lable(ground_truth,hms):
    ## Variable List
    ## ground_truth: The WxH ground truth labels
    ## hms: the hms class object frmo which we extract the shape of the image and the labels
    
    ## Output list
    ## label_arg_max: The most common class in eahc superpixel which is assigned as its label.
    ## label_intersection: the class contents of each superpixel
    
    
    label_intersection = np.zeros((hms.ClusterNumber,np.amax(ground_truth)+1))
    for i in range(hms.Height):
        for j in range(hms.Width):
            label_intersection[hms.Labels[i,j],ground_truth[i,j]] += 1 
            
    label_arg_max = np.zeros((hms.ClusterNumber))
    for i in range(hms.ClusterNumber):
        label_arg_max[i] = np.argmax(label_intersection[i,:])
        
    return label_arg_max,label_intersection
