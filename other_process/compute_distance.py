# -*- coding:utf-8 -*-

import argparse
import numpy as np
from sklearn.cluster import KMeans
import os

import torch
import math

CLASS_NAME = np.array(["aeroplane","bicycle","bird","boat","bottle",
             "bus","car","cat","chair","cow",
             "diningtable","dog","horse","motorbike","person",
             "pottedplant","sheep","sofa","train","tvmonitor"])
FEATURE_RP = "./feature_representation/VOC_2008"

def mkdir(name):
    '''
    Create folder
    '''
    isExists=os.path.exists(name)
    if not isExists:
        os.makedirs(name)

def main(args):
    cluster_centers = []
    for class_name in CLASS_NAME:
        # Get the path
        feature_name = os.path.join(FEATURE_RP, class_name+".npy")
        feature = np.load(feature_name)

        # Cluster
        kmeans = KMeans(n_clusters=args.n_clusters, random_state=0)
        kmeans.fit(feature)

        # Get the cluster centers
        cluster_centers.append(kmeans.cluster_centers_)
        
        del kmeans
    
    confusion_matrix = np.zeros((20,20))
    
    with torch.no_grad():
        for i in range(len(CLASS_NAME)):
            for j in range(len(CLASS_NAME)):
                x_norm = torch.nn.functional.normalize(torch.Tensor(cluster_centers[i]), p=2, dim=1)
                y_norm = torch.nn.functional.normalize(torch.Tensor(cluster_centers[j]), p=2, dim=1)
                similarity = torch.mm(x_norm, y_norm.t())
                similarity = torch.clamp(similarity,-1,1)   # prevent nan
                
                similarity = 1 - torch.arccos(similarity) /  math.pi
                similarity = torch.mean(similarity)
                
                confusion_matrix[i][j] = similarity.item()

                # print("i: {}, j: {}".format(i,j+i+1))
    mkdir("./cluster_txt")
    np.savetxt("./cluster_txt/cluster_"+str(args.n_clusters)+".txt", confusion_matrix)
    mkdir("./cluster_matrix")
    np.save("./cluster_matrix/cluster_"+str(args.n_clusters)+".npy", confusion_matrix)


def parse_args():
    parser = argparse.ArgumentParser(description='VOC 2008 datasets, attributes prediction')
    parser.add_argument('--n-clusters', type=int,
        default=2,
        help='Clustering number.')
    
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)


