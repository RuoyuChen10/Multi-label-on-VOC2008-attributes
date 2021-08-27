# -*- coding:utf-8 -*-

import argparse
import numpy as np
from sklearn.cluster import KMeans
import os

import torch
import math

CLASS_NAME_SPLIT1 = np.array(["aeroplane","bicycle","boat","bottle","car",
                              "cat","chair","diningtable","dog","horse",
                              "person","pottedplant","sheep","train","tvmonitor",
                              "bird","bus","cow","motorbike","sofa"])

CLASS_NAME_SPLIT2 = np.array(["bicycle","bird","boat","bus","car",
                              "cat","chair","diningtable","dog","motorbike","person","pottedplant","sheep","train","tvmonitor",
                              "aeroplane","bottle","cow","horse","sofa"])

CLASS_NAME_SPLIT3 = np.array(["aeroplane","bicycle","bird","bottle","bus",
                              "car","chair","cow","diningtable","dog",
                              "horse","person","pottedplant","train","tvmonitor",
                              "boat","cat","motorbike","sheep","sofa"])

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
    # Choose class index
    if args.split == "split1":
        CLASS_NAME = CLASS_NAME_SPLIT1
    elif args.split == "split2":
        CLASS_NAME = CLASS_NAME_SPLIT2
    elif args.split == "split3":
        CLASS_NAME = CLASS_NAME_SPLIT3
    
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
    mkdir(os.path.join("./cluster_txt/", args.split))
    np.savetxt(os.path.join("./cluster_txt/", args.split) + "/cluster_"+str(args.n_clusters)+".txt", confusion_matrix)
    mkdir(os.path.join("./cluster_matrix", args.split))
    np.save(os.path.join("./cluster_matrix", args.split) + "/cluster_"+str(args.n_clusters)+".npy", confusion_matrix)


def parse_args():
    parser = argparse.ArgumentParser(description='VOC 2008 datasets, attributes relationship.')
    parser.add_argument('--split', type=str,
        default="split1",
        choices=["split1","split2","split3"],
        help='which set.')
    parser.add_argument('--n-clusters', type=int,
        default=7,
        help='Clustering number.')
    
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)


