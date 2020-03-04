import numpy as np
import matplotlib.pyplot as plt

def node_label_full(ground_truth,hms):    
    label_intersection = np.zeros((hms.ClusterNumber,np.amax(ground_truth)+1))
    for i in range(hms.Height):
        for j in range(hms.Width):
            label_intersection[hms.Labels[i,j],ground_truth[i,j]] += 1    
    return label_intersection

# Python code to remove duplicate elements
def Remove(duplicate):
    final_list = []
    for num in duplicate:
        if num not in final_list:
            final_list.append(num.copy())           
    for i in range(len(final_list)):
        final_list[i].append(duplicate.count(final_list[i]));
    return final_list

def sparseness_operator(ground_truth,number_of_samples,hms,spectral_original):
    sparse_ground_truth =  np.reshape(np.copy(ground_truth) , -1)    
    # HOW MANY OF EACH LABEL DO WE WANT 
    number_of_classes = np.amax(ground_truth)
    for i in range(number_of_classes):
        index = np.where(sparse_ground_truth == i+1)[0]
        if(index.shape[0] < number_of_samples):
            index = np.random.choice(index,index.shape[0],replace = False)
        else:
            index = np.random.choice(index,index.shape[0] - number_of_samples,replace = False)
        index = np.sort(index)
        sparse_ground_truth[index] = 0    
    sparse_ground_truth = sparse_ground_truth.reshape((ground_truth.shape))        
    return sparse_ground_truth    
    
def node_label_initialisation(sparse_ground_truth,hms):
    
    label_intersection = np.zeros((hms.ClusterNumber,np.amax(sparse_ground_truth)))
    label_probs = np.zeros((hms.ClusterNumber,np.amax(sparse_ground_truth)))
    
    for i in range(hms.Height):
        for j in range(hms.Width):
            if(sparse_ground_truth[i,j] != 0):
                label_intersection[hms.Labels[i,j],sparse_ground_truth[i,j]-1] += 1 
             
    for i in range(hms.ClusterNumber):
        if(np.sum(label_intersection[i,:]) > 0):
            label_probs[i,:] =  label_intersection[i,:] / np.sum(label_intersection[i,:])

    return label_probs,label_intersection
   
def initial_accuracy_assessment(ground_truth,cluster_labels,node_labels):
    count_in = 0
    count_out = 0
    accuracy = 0
    
    for i in range(ground_truth.shape[0]):
        for j in range(ground_truth.shape[1]):            
            if(ground_truth[i,j] != 0):
                count_out += 1 
                if(node_labels[cluster_labels[i,j]] != 0):
                    count_in += 1 
                    if(node_labels[cluster_labels[i,j]] == ground_truth[i,j]):
                        accuracy += 1                
    
    #print("Accuracy", accuracy/count_in , "Over percentage" , count_in/count_out , "Pixels labeled" , count_in)
    accuracy = accuracy/count_in
    coverage = count_in/count_out    
    return accuracy, coverage, count_in    

def produce_classification(node_labels,labels,ground_truth):
    pro_class = np.zeros(labels.shape,dtype=int)
    for i in range(pro_class.shape[0]):
        for j in range(pro_class.shape[1]):
            pro_class[i,j] = node_labels[labels[i,j]]
            
    return pro_class

def spreading_accuracy(new_ground_truth,ground_truth):
    accuracy = 0
    count = 0 
    false_positives = 0
    for i in range(new_ground_truth.shape[0]):
            for j in range(new_ground_truth.shape[1]):
                if(new_ground_truth[i,j] != 0 and ground_truth[i,j] != 0):
                    count += 1
                    if(ground_truth[i,j] == new_ground_truth[i,j]):
                        accuracy += 1
                if(new_ground_truth[i,j] != 0 and ground_truth[i,j] ==0):
                    false_positives += 1
                    
    print("Count" , count , "Accuracy" , accuracy/count , "False Positives" , false_positives)
    
    
def classification_image_mapping(pro_class,ground_truth,name,OA):
    
    if(name == 'Salinas'):
        colours = np.zeros((17,3),int)
        image = np.zeros((pro_class.shape[0],pro_class.shape[1],3))
        
        colours[0,:]  = [0,0,0]
        colours[1,:]  = [140,67,46]
        colours[2,:]  = [0,0,255]
        colours[3,:]  = [255,100,0]
        colours[4,:]  = [0,255,123]
        colours[5,:]  = [164,75,155]
        colours[6,:]  = [101,174,255]
        colours[7,:]  = [118,254,172]
        colours[8,:]  = [60,91,112]
        colours[9,:]  = [255,255,0]
        colours[10,:] = [255,255,125]
        colours[11,:] = [255,0,255]
        colours[12,:] = [100,0,255]
        colours[13,:] = [0,172,254]
        colours[14,:] = [0,255,0]
        colours[15,:] = [171,175,80]
        colours[16,:] = [101,193,60]
        
        colours = colours/255
        for i in range(pro_class.shape[0]):
            for j in range(pro_class.shape[1]):
                image[i,j,:] = colours[pro_class[i,j],:]
                
                
        pro_class_2 = np.copy(pro_class)
        for i in range(pro_class.shape[0]):
            for j in range(pro_class.shape[1]):
                if(ground_truth[i,j] == 0):
                    pro_class_2[i,j] = 0
                    
        image2 = np.zeros((pro_class.shape[0],pro_class.shape[1],3))
        for i in range(pro_class.shape[0]):
            for j in range(pro_class.shape[1]):
                image2[i,j,:] = colours[pro_class_2[i,j],:]
                
        plt.imsave('full_class_map_sal_' + str(OA) + '.png',image)
        plt.imsave('nb_class_map_sal_' + str(OA) + '.png',image2)

    if(name == 'Indian'):
        colours = np.zeros((17,3),int)
        image = np.zeros((pro_class.shape[0],pro_class.shape[1],3))
        
        colours[0,:]  = [0,0,0]
        colours[1,:]  = [140,67,46]
        colours[2,:]  = [0,0,255]
        colours[3,:]  = [255,100,0]
        colours[4,:]  = [0,255,123]
        colours[5,:]  = [164,75,155]
        colours[6,:]  = [101,174,255]
        colours[7,:]  = [118,254,172]
        colours[8,:]  = [60,91,112]
        colours[9,:]  = [255,255,0]
        colours[10,:] = [255,255,125]
        colours[11,:] = [255,0,255]
        colours[12,:] = [100,0,255]
        colours[13,:] = [0,172,254]
        colours[14,:] = [0,255,0]
        colours[15,:] = [171,175,80]
        colours[16,:] = [101,193,60]
        
        colours = colours/255
        for i in range(pro_class.shape[0]):
            for j in range(pro_class.shape[1]):
                image[i,j,:] = colours[pro_class[i,j],:]
                
                
        pro_class_2 = np.copy(pro_class)
        for i in range(pro_class.shape[0]):
            for j in range(pro_class.shape[1]):
                if(ground_truth[i,j] == 0):
                    pro_class_2[i,j] = 0
                    
        image2 = np.zeros((pro_class.shape[0],pro_class.shape[1],3))
        for i in range(pro_class.shape[0]):
            for j in range(pro_class.shape[1]):
                image2[i,j,:] = colours[pro_class_2[i,j],:]
                
        plt.imsave('full_class_map_ind_' + str(OA) + '.png',image)
        plt.imsave('nb_class_map_ind_' + str(OA) + '.png',image2)
        
        
    if(name == 'PaviaU'):
        colours = np.zeros((10,3),int)
        image = np.zeros((pro_class.shape[0],pro_class.shape[1],3))
        
        colours[0,:]  = [0,0,0]
        colours[1,:]  = [192,192,192]
        colours[2,:]  = [0,255,0]
        colours[3,:]  = [0,255,255]
        colours[4,:]  = [0,128,0]
        colours[5,:]  = [255,0,255]
        colours[6,:]  = [165,82,41]
        colours[7,:]  = [128,0,128]
        colours[8,:]  = [255,0,0]
        colours[9,:]  = [255,255,0]

        
        colours = colours/255
        for i in range(pro_class.shape[0]):
            for j in range(pro_class.shape[1]):
                image[i,j,:] = colours[pro_class[i,j],:]
                
                
        pro_class_2 = np.copy(pro_class)
        for i in range(pro_class.shape[0]):
            for j in range(pro_class.shape[1]):
                if(ground_truth[i,j] == 0):
                    pro_class_2[i,j] = 0
                    
        image2 = np.zeros((pro_class.shape[0],pro_class.shape[1],3))
        for i in range(pro_class.shape[0]):
            for j in range(pro_class.shape[1]):
                image2[i,j,:] = colours[pro_class_2[i,j],:]
                
        plt.imsave('full_class_map_pu_' + str(OA) + '.png',image)
        plt.imsave('nb_class_map_pu_' + str(OA) + '.png',image2)
        
        
        
def where_are_my_errors(labels,pro_class,gt):
    region_errors = np.zeros((np.amax(labels)+1))
    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            if(gt[i,j] != 0):
               if(gt[i,j] != pro_class[i,j]):
                   region_errors[labels[i,j]] += 1
    return region_errors