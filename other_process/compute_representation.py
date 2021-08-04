# -*- coding:utf-8 -*-

import argparse
import os
import numpy as np
import time

import torchvision.transforms as transforms

import torch
import torch.nn as nn 

from PIL import Image

import sys
sys.append("..")

from config import Config
from Logging import Logger
from dataset import Dataset

from collections import OrderedDict
from models.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152

from tqdm import tqdm

TRAIN_PATH   = "/exdata/RuoyuChen/describing_objects_by_their_attributes/attribute_data/apascal_train.txt"
TEST_PATH    = "/exdata/RuoyuChen/describing_objects_by_their_attributes/attribute_data/apascal_test.txt"
IMAGE_PATH = "/exdata/RuoyuChen/describing_objects_by_their_attributes/Pascal_Dataset/JPEGImages"

CLASS_NAME = np.array(["aeroplane","bicycle","bird","boat","bottle",
             "bus","car","cat","chair","cow",
             "diningtable","dog","horse","motorbike","person",
             "pottedplant","sheep","sofa","train","tvmonitor",
             "donkey","monkey","goat","wolf","jetski",
             "zebra","centaur","mug","statue","building",
             "bag","carriage"])

def mkdir(name):
    '''
    Create folder
    '''
    isExists=os.path.exists(name)
    if not isExists:
        os.makedirs(name)

def _get_features_hook(module, input, output):
        global hook_feature 
        hook_feature = output.view(output.size(0), -1)[0]
        # print("feature shape:{}".format(hook_feature.size()))

def _register_hook(net,layer_name):
    for (name, module) in net.named_modules():
        if name == layer_name:
            module.register_forward_hook(_get_features_hook)

def Load_model(network_name, num_classes, attribute_classes, strategy, pretrained):
    if network_name == "ResNet50":
        model = ResNet50(num_classes, attribute_classes, strategy)
    elif network_name == "ResNet101":
        model = ResNet101(num_classes, attribute_classes, strategy)

    assert os.path.exists(pretrained)
    model_dict = model.state_dict()
    pretrained_param = torch.load(pretrained)

    new_state_dict = OrderedDict()
    # pretrained_dict = {k: v for k, v in pretrained_param.items() if k in model_dict}
    for k, v in pretrained_param.items():
        if k in model_dict:
            new_state_dict[k] = v
            print("Load layer {}".format(k))
        elif k[7:] in model_dict:
            new_state_dict[k[7:]] = v
            print("Load layer {}".format(k[7:]))
    
    model.load_state_dict(new_state_dict)
    
    print("\033[32m{} ".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())) + 
                 "\033[0mLoad model" + 
                 "\033[34m {} ".format(network_name) + 
                 "\033[0mfrom pretrained" + 
                 "\033[34m {}".format(pretrained))

    return model

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device
    device = torch.device("cuda")

    # Configuration file
    cfg = Config(args.configuration_file)

    model = Load_model(cfg.BACKBONE, cfg.CLASS_NUM, cfg.ATTRIBUTE_NUM, cfg.STRATEGY, args.Test_model)
    model.to(device)
    model.eval()

    # Hook
    _register_hook(model,"avgpool")

    transformser = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
            ])

    feature_map = [[] for i in range(20)]
    
    with torch.no_grad():
        for line in open(TEST_PATH):
            str = line.rstrip("\n")
            splits = str.split(' ')

            class_name = splits[1]
            index = np.where(CLASS_NAME==class_name)
            
            image_path = os.path.join(IMAGE_PATH, splits[0])
            data = Image.open(image_path)
            data = data.crop((int(splits[2]),int(splits[3]),int(splits[4]),int(splits[5])))
            data = transformser(data)
            
            model(torch.unsqueeze(data, 0).to(device))
            feature = hook_feature
            feature_map[index[0][0]].append(feature.cpu().numpy())
    
    for i in range(20):
        np.save(os.path.join(args.save_path, CLASS_NAME[i]+'.npy'),np.array(feature_map[i]))

def parse_args():
    parser = argparse.ArgumentParser(description='VOC 2008 datasets, attributes prediction')
    parser.add_argument('--configuration-file', type=str,
        default='../configs/Base-ResNet101-B.yaml',
        help='The model configuration file.')
    parser.add_argument('--gpu-device', type=str, default="1",
                        help='GPU device')
    parser.add_argument('--Test-model', type=str, 
    # default="./checkpoint/backbone-item-epoch-990.pth",
    default="../checkpoint/Save_ckpt_1/backbone-item-epoch-500.pth",
                        help='Model weight for testing.')
    parser.add_argument('--save-path', type=str, default="./feature_representation/VOC_2008",
                        help='Save path')
    args = parser.parse_args()

    return args
    
if __name__ == "__main__":
    args = parse_args()
    mkdir(args.save_path)
    main(args)