import numpy as np
from numpy import matlib
import numexpr as ne 


### Function that performs dimensionality reduction using the maximum noise fraction method
def dimensional_reduction_mnf(spectral_data,number_of_components):    
    ## Variable List
    ## spectral_data: The input image
    ## number_of_components: the requested number of components

    ## Output List
    ## RD_img: Reduced image   
    [height,width,bands] = spectral_data.shape
    spectral_data_2d = np.reshape(spectral_data,(height*width,bands))
    
    ## Centre the data 
    centre_data = spectral_data_2d - np.ones((height*width,1)) * np.reshape(np.average(spectral_data_2d,axis=0),(1,bands)) 
    
    ## Calculate the covariance matrix for the whole data
    sigma_x = (1/(height*width - 1)) * np.linalg.multi_dot((np.transpose(centre_data),centre_data))
    
    ## Estimate the covariance matrix of the noise 
    D = np.zeros(np.shape(spectral_data))
    for i in range(1,height):
        D[i,:,:] = (spectral_data[i,:,:] - spectral_data[i-1,:,:])/2
    for i in range(0,width-1):
        D[:,i,:] += (spectral_data[:,i,:] - spectral_data[:,i+1,:])/2

    ## Centre the data 
    D_2d = np.reshape(D,(height*width,bands))
    centre_data_d = D_2d - np.ones((height*width,1)) * np.reshape(np.average(D_2d,axis=0),(1,bands)) 
    sigma_n = (1/(height*width - 1)) * np.linalg.multi_dot((np.transpose(centre_data_d),centre_data_d))
    
    eig_m =  np.linalg.multi_dot((np.linalg.inv(sigma_n),sigma_x))
    eigenvalues,eigenvectors = np.linalg.eig(eig_m)
    project_H = eigenvectors[:,:number_of_components]
    
    RD_img_mat = np.linalg.multi_dot((spectral_data_2d,project_H))
    RD_img = np.reshape(RD_img_mat,(height,width,number_of_components))
    
    return RD_img

# %%

### Function that create the covariance matrices for clustering
def create_logm_matrices(spectral_mnf,window_size,K):    
    ## Variable List
    ## spectral_mnf: The reduced image
    ## window_size: the window size focused on each pixel
    ## K: The number of neighbours
    
    ## Output List
    ## all_matrices: The lcmr matrices in the shape dxdxhxw 
    
# %%
    tol = 0.001
    window_size = 25
    K = 400
    size_0 = spectral_mnf.shape[0]
    size_1 = spectral_mnf.shape[1]
    size_2 = spectral_mnf.shape[2]
    scale = int(np.floor(window_size/2))
    d = size_2
    
    spectral_extended = np.pad(spectral_mnf,((scale,scale),(scale,scale),(0,0)),'symmetric')
    i_d = int(np.ceil(window_size*window_size/2))    
    all_matrices = np.zeros((d,d,size_0,size_1))   
# %%
    
    for i in range(size_0):
# %%
        pp = i+(2*scale)+1
        tt_idea = spectral_extended[i:pp,:,:] 
        tt_big_2d = np.reshape(tt_idea ,(window_size*tt_idea.shape[1],d),order='F')
        tt_norms_big = np.linalg.norm(tt_big_2d,axis=1)
        tt_norms_big_repmat = np.transpose(np.matlib.repmat(tt_norms_big,d,1))
        big_divide = ne.evaluate('tt_big_2d/tt_norms_big_repmat')
        #big_divide = np.ascontiguousarray(np.divide(tt_big_2d,tt_norms_big_repmat))
 # %%       
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
# %%           
            tmp = np.dot(np.transpose(centered_mat),centered_mat) 
            tmp = np.divide(tmp,K-1) 
            tmp_tmp = tmp + tol*np.identity(d)*np.trace(tmp)           
            [s,u] = np.linalg.eigh(tmp_tmp)     
            hmm = np.diag(ne.evaluate("log(s)"))
            all_matrices[:,:,i,j]  = np.linalg.multi_dot((u,hmm,np.linalg.inv(u)))   
            
            # %% 
    return all_matrices


