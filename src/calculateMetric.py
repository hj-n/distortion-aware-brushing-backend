from sklearn import cluster
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, completeness_score, fowlkes_mallows_score, homogeneity_completeness_v_measure, homogeneity_score
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score, silhouette_score, v_measure_score

from sklearn.cluster import k_means, AgglomerativeClustering
import numpy as np
import json
from tqdm import tqdm
from .helpers import dunn, snndpc

'''
Get a point set and return how the separability score between A and B
'''
def check_separability(
	points, 
    class_label,
    clustered_label,
    label_num
):

    return_dict = {}

    ami_result = ami(class_label, clustered_label)
    return_dict["ami"] = ami_result
    arand_result = arand(class_label, clustered_label)
    return_dict["arand"] = arand_result
    if(label_num >= 2):
        calinski_result = calinski(points, clustered_label)
        return_dict["calinski"] = calinski_result
        davis_result = davis(points, clustered_label)
        return_dict["davis"] = davis_result
        silhouette_result = silhouette(points, clustered_label)
        return_dict["silhouette"] = silhouette_result
    completeness_result = completeness(class_label, clustered_label)
    return_dict["completeness"] = completeness_result
    fowlkes_result = fowlkes(class_label, clustered_label)
    return_dict["fowlkes"] = fowlkes_result
    homogeneity_result = homogeneity(class_label, clustered_label)
    return_dict["homogeneity"] = homogeneity_result
    mi_result = mi(class_label, clustered_label)
    return_dict["mi_result"] = mi_result
    normmi_result = normmi(class_label, clustered_label)
    return_dict["normmi"] = normmi_result
    #rc_result = rc(class_label, clustered_label)
    #return_dict["rc"] = rc_result
    vm_result = vm(class_label, clustered_label)
    return_dict["vm"] = vm_result


    return return_dict


'''
Metrics that measures the separability of classes itself
1. Silhouette Coefficeint
'''

def calinski(points, labels): #score has no bound
    return calinski_harabasz_score(points, labels)

def davis(points, labels):
    return davies_bouldin_score(points, labels)

def silhouette(points, labels):
	return (silhouette_score(points, labels) + 1) / 2



'''
Metrics that compares the ground truth with clustering results
1. adjusted mutual info score
2. adjusted rand score
'''

def ami(labels_true, labels_pred):
	return adjusted_mutual_info_score(labels_true, labels_pred)

def arand(labels_true, labels_pred):
	return adjusted_rand_score(labels_true, labels_pred)

def completeness(labels_true, labels_pred):
	return completeness_score(labels_true, labels_pred)

def fowlkes(labels_true, labels_pred):
    return fowlkes_mallows_score(labels_true, labels_pred)

def homogeneity(labels_true, labels_pred):
	return homogeneity_score(labels_true, labels_pred)

def mi(labels_true, labels_pred):
    return mutual_info_score(labels_true, labels_pred)

def normmi(labels_true, labels_pred):
    return normalized_mutual_info_score(labels_true, labels_pred)

#def rc(labels_true, labels_pred):
#   return rand_score(labels_true, labels_pred)

def vm(labels_true, labels_pred):
    return v_measure_score(labels_true, labels_pred)

def clusteredMetric(
	points,
    class_label,
    clustered_label,
    label_num
):
    file_path_label = "./userstudy/dataset_method_samplerate_clustered_label.json"
    file_path_result = "./userstudy/dataset_method_samplerate_metric_result.json"
    label_data = clustered_label.tolist()

    selected_points = np.nonzero(clustered_label)[0]
    points = points[selected_points]
    class_label = class_label[selected_points]
    clustered_label = clustered_label[selected_points]
    clustermetric_result = check_separability(points, class_label, clustered_label, label_num)
    print(clustermetric_result)

    with open(file_path_label, 'w') as outfile:
        json.dump(label_data, outfile)
    with open(file_path_result, 'w') as outfile:
        json.dump(clustermetric_result, outfile)

'''
TEST CODE


cluster_A = np.random.rand(20, 10)
cluster_B = np.random.rand(20, 10)

a, b = check_separability(cluster_A, cluster_B, ["silhouette",  "kmeans", "dpc", "agglomerative"])

print(a)
print(b)
'''