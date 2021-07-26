# -*- coding: utf-8 -*-  

"""
Created on 2021/07/14

@author: Ruoyu Chen
"""

import os
import random
import numpy as np

import torchvision.transforms as transforms

from PIL import Image
from torch.utils import data

class Dataset(data.Dataset):
    """
    Read datasets

    Args:
        dataset_root: the images dir path
        dataset_list: the labels
    """
    def __init__(self, dataset_root, dataset_list, class_name, strategy, data_type="train"):
        self.class_name = class_name
        self.class_num = len(class_name)
        self.strategy = strategy

        with open(dataset_list,"r") as file:
            datas = file.readlines()

        data = [os.path.join(dataset_root, data_.rstrip("\n")) for data_ in datas]

        
        if data_type == "train":
            self.data = data
            # self.data = np.random.permutation(data)
            self.transforms = transforms.Compose([
                # transforms.RandomHorizontalFlip(),
                # transforms.CenterCrop(196),
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
            ])
        elif data_type == "test":
            self.data = data
            self.transforms = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self,index):
        # Sample
        sample = self.data[index]
        
        # data and label information
        splits = sample.split(' ')
        image_path = splits[0]

        data = Image.open(image_path)
        data = data.crop((int(splits[2]),int(splits[3]),int(splits[4]),int(splits[5])))

        data = self.transforms(data)
        
        class_label = np.int32(
            self.class_name.index(splits[1])
        )
        attribute_label = [int(x) for x in splits[6:]]

        if self.strategy == "A":
            label = self.label_convert(class_label,attribute_label)
        elif self.strategy == "B":
            label = np.array(attribute_label).astype(np.float32)

        return data.float(), label
    
    def label_convert(self,class_label,attribute_label):
        """
        The input label convert to the network label
        """
        label = []
        for attribute in attribute_label:
            if attribute == 1:
                label.append(class_label)
            elif attribute == 0:
                label.append(self.class_num)
        return np.array(label).astype(np.long)
