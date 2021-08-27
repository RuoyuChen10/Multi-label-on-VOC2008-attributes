# -*- coding:utf-8 -*-

import argparse
import os
import numpy as np
import math
import torch
from tqdm import tqdm

TRAIN_PATH   = "/exdata/RuoyuChen/describing_objects_by_their_attributes/attribute_data/apascal_train.txt"
TEST_PATH    = "/exdata/RuoyuChen/describing_objects_by_their_attributes/attribute_data/apascal_test.txt"

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

def mkdir(name):
    '''
    Create folder
    '''
    isExists=os.path.exists(name)
    if not isExists:
        os.makedirs(name)

def main(args):
    if args.split == "split1":
        CLASS_NAME = CLASS_NAME_SPLIT1
    elif args.split == "split2":
        CLASS_NAME = CLASS_NAME_SPLIT2
    elif args.split == "split3":
        CLASS_NAME = CLASS_NAME_SPLIT3

    data = np.zeros((20, 64))
    data_num = np.zeros(20)
    for line in tqdm(open(TRAIN_PATH)):
        str = line.rstrip("\n")
        splits = str.split(' ')
        attribute = np.array([int(x) for x in splits[6:]])

        class_name = splits[1]
        index = np.where(CLASS_NAME==class_name)

        data[index] += attribute
        data_num[index] +=1
    # Ground truth feature
    feature = data#/data_num.reshape((-1,1))
    
    confusion_matrix = np.zeros((20,20))

    with torch.no_grad():
        for i in range(len(CLASS_NAME)):
            for j in range(len(CLASS_NAME)):
                x_norm = torch.nn.functional.normalize(torch.Tensor(feature[i]), p=2, dim=0)
                y_norm = torch.nn.functional.normalize(torch.Tensor(feature[j]), p=2, dim=0)

                similarity = torch.mm(x_norm.reshape((1,-1)), y_norm.reshape((-1,1)))
                similarity = torch.clamp(similarity,-1,1)   # prevent nan
                
                similarity = 1 - torch.arccos(similarity) /  math.pi
                similarity = torch.mean(similarity)
                
                confusion_matrix[i][j] = similarity.item()

                # print("i: {}, j: {}".format(i,j+i+1))
    mkdir(os.path.join("./cluster_txt/", args.split))
    np.savetxt(os.path.join("./cluster_txt/", args.split) + "/knowledge_gt.txt", confusion_matrix)
    mkdir(os.path.join("./cluster_matrix", args.split))
    np.save(os.path.join("./cluster_matrix", args.split) + "/knowledge_gt.npy", confusion_matrix)

def parse_args():
    parser = argparse.ArgumentParser(description='VOC 2008 datasets, attributes relationship.')
    parser.add_argument('--split', type=str,
        default="split1",
        choices=["split1","split2","split3"],
        help='which set.')
    parser.add_argument('--n-clusters', type=int,
        default=1,
        help='Clustering number.')
    
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)