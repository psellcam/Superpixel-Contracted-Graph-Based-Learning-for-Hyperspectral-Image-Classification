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
def connectivity_cython(long[:,::1] segments,int min_size, int max_size):
    """ Helper function to remove small disconnected regions from the labels
    Parameters
    ----------
    segments : 2D array of int, shape (Y, X)
        The label field/superpixels found by SLIC.
    min_size: int
        Minimum size of the segment
    max_size: int
        Maximum size of the segment. This is done for performance reasons,
        to pre-allocate a sufficiently large array for the breadth first search
    Returns
    -------
    connected_segments : 3D array of int, shape (Z, Y, X)
        A label field with connected labels starting at label=1
    """

    # get image dimensions
    cdef int height, width
    height = segments.shape[0]
    width = segments.shape[1]

    # neighborhood arrays
    cdef long[::1] ddx = np.array((1, -1, 0, 0), dtype=int)
    cdef long[::1] ddy = np.array((0, 0, 1, -1), dtype=int)
    
    # new object with connected segments initialized to -1
    cdef long[:, ::1] connected_segments = -1 * np.ones_like(segments, dtype=int)

    cdef int current_new_label = 0
    cdef int label = 0

    # variables for the breadth first search
    cdef int current_segment_size = 1
    cdef int bfs_visited = 0
    cdef int adjacent

    cdef int yy, xx

    cdef long[:, ::1] coord_list = np.zeros((max_size, 2), dtype=int)
    
    cdef int i,x,y
  
    with nogil:
        for y in range(height):
            for x in range(width):
                if connected_segments[y, x] >= 0:
                    continue
                # find the component size
                adjacent = 0
                label = segments[y, x]
                connected_segments[y, x] = current_new_label
                current_segment_size = 1
                bfs_visited = 0
                coord_list[bfs_visited, 0] = y
                coord_list[bfs_visited, 1] = x
                
                #perform a breadth first search to find
                # the size of the connected component
                               
                while bfs_visited < current_segment_size < max_size:
                    for i in range(4):
                        yy = coord_list[bfs_visited, 0] + ddy[i]
                        xx = coord_list[bfs_visited, 1] + ddx[i]
                        if (0 <= xx < width and 0 <= yy < height):
                            if (segments[yy, xx] == label and connected_segments[yy, xx] == -1):
                                connected_segments[yy, xx] = current_new_label
                                coord_list[current_segment_size, 0] = yy
                                coord_list[current_segment_size, 1] = xx
                                current_segment_size += 1
                                if current_segment_size >= max_size:
                                    break
                            elif (connected_segments[yy, xx] >= 0 and
                                  connected_segments[yy, xx] != current_new_label):
                                adjacent = connected_segments[yy, xx]
                    bfs_visited += 1

                # change to an adjacent one, like in the original paper
                if current_segment_size < min_size:
                    for i in range(current_segment_size):
                        connected_segments[coord_list[i, 0], coord_list[i, 1]] = adjacent
                else:
                    current_new_label += 1

    return np.asarray(connected_segments)

@cython.boundscheck(False)
def assign_clusters_cython(double[:, :, ::1] imageGrid, int gridStep, int cluster_number,
                           double[:,::1] distanceArray, double[:,::1] clusterPositions, double[:,::1] clusterColours, 
                           double[:,:,::1] clusterLCMR, double [::1] clusterScale, double compactness, double[:,:,:,::1] lcmr, double a_1, double a_2):    
    
    cdef int i,c,h,w,h_min,h_max,w_min,w_max,height, weight,bands,lcmr_bands,ii,jj
    cdef double Dc,Ds,D,ch,cw,dh,dw,D_lcmr
    cdef int adaptiveGridStep      
    
    height = imageGrid.shape[0]
    width = imageGrid.shape[1]
    bands = imageGrid.shape[2]    
    lcmr_bands = clusterLCMR.shape[0]
    
    cdef double[:, ::1] lcmr_difference = np.zeros((lcmr_bands, lcmr_bands))
    
    
    # Storing the labels
    cdef long[:,::1] labels = -1*np.ones((height,width),dtype='int')
    
    with nogil:       
        for i in range(cluster_number):
            
            ch = clusterPositions[i,0]
            cw = clusterPositions[i,1]
                               
            adaptiveGridStep =  int(round(clusterScale[i]*gridStep))                
            h_min = <int>max(ch - adaptiveGridStep, 0)
            h_max = <int>min(ch + adaptiveGridStep + 1, height)
            w_min = <int>max(cw - adaptiveGridStep, 0)
            w_max = <int>min(cw + adaptiveGridStep + 1, width)      
                    
                    
            for h in range(h_min,h_max):
                dh = (ch - h) ** 2      
                for w in range(w_min, w_max):
                    dw = (cw - w) ** 2 
                    Ds = sqrt(dw+dh)
                    
                    Dc = 0
                    D_lcmr = 0
                    
                    for ii in range(lcmr_bands):
                        for jj in range(lcmr_bands):
                            lcmr_difference[ii,jj] = clusterLCMR[ii,jj,i] - lcmr[ii,jj,h,w]
                            D_lcmr += lcmr_difference[ii,jj] * lcmr_difference[ii,jj]
                    
                    D_lcmr = sqrt(D_lcmr)
                    
                    for c in range(bands):
                        Dc += (imageGrid[h,w,c] - clusterColours[i,c]) ** 2                         
                    Dc = sqrt(Dc)
                    
                    D = a_1*Dc + a_2*D_lcmr + (compactness/gridStep)*Ds                    
                    if D < distanceArray[h, w]:                            
                        distanceArray[h, w] = D
                        labels[h,w] = i                            
                                                        
    return labels,distanceArray        
            
    
@cython.boundscheck(False)   
def update_cluster_cython(double[:, :, ::1] imageGrid, double[:,:,:, ::1] lcmr, long[:, ::1] Label, int length, double[:,::1] areas): 
    
    cdef int bands = imageGrid.shape[2]
    cdef int lcmr_bands = lcmr.shape[0]
    
    # Format for data cluster information will be h,w,L,a,b,n
    cdef double[:, ::1] kvector_info = np.zeros((3+bands,length))    
    cdef double[:, ::1] kvector_number = np.zeros((1,length))    
    cdef double[:, :, ::1] kvector_lcmr = np.zeros((lcmr_bands,lcmr_bands,length))   
    
    cdef int i,j,k,locallabel,b_i,ii,jj
        
    cdef int height,width
    cdef double number
    
    height = Label.shape[0]
    width = Label.shape[1]
    
    with nogil:
        for i in range(height):
                for j in range(width):
                    locallabel = Label[i,j]
                    kvector_info[0,locallabel] += i
                    kvector_info[1,locallabel] += j
                    
                    for ii in range(lcmr_bands):    
                        for jj in range(lcmr_bands):
                            kvector_lcmr[ii,jj,locallabel] += lcmr[ii,jj,i,j] 
                            
                    for b_i in range(2,2+bands):                    
                        kvector_info[b_i,locallabel] += imageGrid[i,j,b_i-2]                       
                        
                    kvector_info[2+bands,locallabel] += areas[i,j]
                    kvector_number[0,locallabel] += 1   
                    
        for k in range(length):                
                number = kvector_number[0,k]
                if(number != 0):
                    kvector_info[0,k] = kvector_info[0,k] / number
                    kvector_info[0,k] = round(kvector_info[0,k])
                    kvector_info[1,k] = kvector_info[1,k] / number
                    kvector_info[1,k] = round(kvector_info[1,k])
                    
                    for b_i in range(2,2+bands): 
                        kvector_info[b_i,k] = kvector_info[b_i,k] / number              

                    for ii in range(lcmr_bands):    
                        for jj in range(lcmr_bands):
                            kvector_lcmr[ii,jj,k] = kvector_lcmr[ii,jj,k]/number
                            
                else:
                    kvector_info[0,k] = 10000
                    kvector_info[1,k] = 10000
                    for b_i in range(2,2+bands): 
                        kvector_info[b_i,k] = 0   
                    for ii in range(bands):    
                        for jj in range(bands):
                            kvector_lcmr[ii,jj,k] = 0 
            
    return kvector_info , kvector_lcmr


# Function to calculate the adaptive search range of each cluster seed
@cython.boundscheck(False) 
def find_area_search_limit(int ch, int cw , int S ,double[:,::1] areas ):            
    cdef double area_sum = 0 
    cdef int h_min,h_max,w_min,w_max,h,w,height,width
    
    height = areas.shape[0]
    width = areas.shape[1]
    
               
    
    with nogil:
        h_min = <int>max(ch - S, 0)
        h_max = <int>min(ch + S + 1, height)
        w_min = <int>max(cw - S, 0)
        w_max = <int>min(cw + S + 1, width)      
        
        for h in range(h_min,h_max):
            for w in range(w_min, w_max):
                area_sum += areas[h,w]
                
    return area_sum
                

# Cython equivilant of numpy where for clusters 
@cython.boundscheck(False)
def where_cluster(long[: , ::1] labels,int cluster_number):
    cdef int i,j,height,width,ll 
    height = labels.shape[0]
    width = labels.shape[1]   
    
    cdef int segement_size = int((40*height*width)/cluster_number)    
    cdef long[::1] count = np.zeros((cluster_number),dtype=int)    
    cdef long[:,:,::1] positions = np.zeros((cluster_number,segement_size,2),dtype = int)
    
    

    with nogil:
        for i in range(height):
            for j in range(width): 
                    ll = labels[i,j]                    
                    positions[ll,count[ll],0] = i
                    positions[ll,count[ll],1] = j
                    count[ll] += 1 
    return positions , count
    

# Find all the neighbouring unvisited clusters 
@cython.boundscheck(False)
def neighbour_search(long[:,::1] indexs,long[:,::1] labels,long[::1] visited,int S):      
    cdef long[::1] dx = np.array((1 , -1 , 0 , 0),dtype = int)
    cdef long[::1] dy = np.array((0 , 0 , 1 , -1),dtype = int)       
    cdef int i,j,number,search_i,search_j,s_label,height,width,count,s_i,s_j    
    
    height = labels.shape[0]
    width = labels.shape[1]
    number = indexs.shape[0] 
    count = 0
    cdef long[::1] neighbours = -1*np.ones((2*height + 2*width),dtype = int)  
    
    for i in range(number):
        search_i = indexs[i,0]
        search_j = indexs[i,1]       
        # It has to be in the image
        if( search_i > 0 and search_i < height-1 and search_j > 0 and search_j < width-1):           
            for j in range(4):
                s_i = search_i + dx[j]
                s_j = search_j + dy[j]
                s_label = labels[s_i,s_j]
                # If it isn't visited and the label hasnt been seen before add it
                if(visited[s_label] == -1):                                  
                    neighbours[count] = s_label
                    count += 1                   

    return neighbours,count

# Calculate area element 
@cython.boundscheck(False)
def area_element(double[:,:,::1] image,int m , int S):
    cdef int height,width,bands,i,j,k    
    height = image.shape[0]
    width = image.shape[1]
    bands = image.shape[2]
    cdef double[:, ::1] manifold_areas = np.zeros((height,width))   
    
    
    cdef double[::1] c_p1 = np.zeros((bands)) 
    cdef double[::1] c_p2 = np.zeros((bands)) 
    cdef double[::1] c_p3 = np.zeros((bands)) 
    cdef double[::1] c_p4 = np.zeros((bands)) 
    
    cdef double[::1] c_p2_p1 = np.zeros((bands)) 
    cdef double[::1] c_p2_p3 = np.zeros((bands)) 
    cdef double[::1] c_p4_p3 = np.zeros((bands)) 
    cdef double[::1] c_p4_p1 = np.zeros((bands))        
    
    
    cdef double norm_21,norm_32,norm_43,norm_41,cos_theta,sin_theta,area
    cdef double d_2121,d_2323,d_2123,d_4343,d_4141,d_4341
    
    
    for i in range(1,height-1):
        for j in range(1,width-1):
            d_2121 = 0
            d_2323 = 0 
            d_2123 = 0 
            d_4343 = 0 
            d_4141 = 0 
            d_4341 = 0                    
            
            for k in range(bands):
                c_p1[k] = (image[i,j-1,k] + image[i-1,j-1,k] + image[i-1,j,k] + image[i,j,k])/(4*m)
                c_p2[k] = (image[i,j-1,k] + image[i+1,j-1,k] + image[i+1,j,k] + image[i,j,k])/(4*m)
                c_p3[k] = (image[i,j+1,k] + image[i+1,j+1,k] + image[i+1,j,k] + image[i,j,k])/(4*m)
                c_p4[k] = (image[i,j+1,k] + image[i-1,j+1,k] + image[i-1,j,k] + image[i,j,k])/(4*m)  
                                
            for k in range(bands):                           
                c_p2_p1[k] = c_p1[k]-c_p2[k]
                c_p2_p3[k] = c_p3[k]-c_p2[k]   
                c_p4_p3[k] = c_p3[k]-c_p4[k]  
                c_p4_p1[k] = c_p1[k]-c_p4[k]        
                
            for k in range(bands):
                d_2121 += c_p2_p1[k]*c_p2_p1[k]
                d_2323 += c_p2_p3[k]*c_p2_p3[k] 
                d_2123 += c_p2_p1[k]*c_p2_p3[k] 
                d_4343 += c_p4_p3[k]*c_p4_p3[k] 
                d_4141 += c_p4_p1[k]*c_p4_p1[k] 
                d_4341 += c_p4_p3[k]*c_p4_p1[k] 
                
        
            norm_21 = sqrt( 1.0/(S*S) +  d_2121)
            norm_23 = sqrt( 1.0/(S*S) +  d_2323) 
            cos_theta = d_2123 / (norm_21 * norm_23)   
            sin_theta = sqrt(1-cos_theta*cos_theta)
            manifold_areas[i,j] = 0.5 * norm_21 * norm_23 * sin_theta
                                
                                
            norm_43 = sqrt( 1.0/(S*S) +  d_4343)
            norm_41 = sqrt( 1.0/(S*S) +  d_4141)   
            cos_theta = d_4341 / (norm_43 * norm_41)
            sin_theta = sqrt(1-cos_theta*cos_theta)
            manifold_areas[i,j] += 0.5 * norm_43 * norm_41 * sin_theta      

    for i in range(height):
        manifold_areas[i,0] = manifold_areas[i,1]
        manifold_areas[i,width-1] = manifold_areas[i,width-2]
    for j in range(width):
        manifold_areas[0,j] = manifold_areas[1,j]
        manifold_areas[height-1,j] = manifold_areas[height-2,j]
        
    return manifold_areas
                        
                        

                        
                        
                        
                        
                        
                        
                        