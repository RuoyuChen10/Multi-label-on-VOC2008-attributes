# -*- coding:utf-8 -*-

import argparse
import os
import numpy as np
import time
import xmltodict

import torchvision.transforms as transforms

import torch
import torch.nn as nn 

from PIL import Image

import sys
sys.path.append("../")

from config import Config
from Logging import Logger
from dataset import Dataset

from collections import OrderedDict
from models.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152

from tqdm import tqdm

CLASS_NAME = np.array(["aeroplane","bicycle","bird","boat","bottle",
             "bus","car","cat","chair","cow",
             "diningtable","dog","horse","motorbike","person",
             "pottedplant","sheep","sofa","train","tvmonitor"])

def _get_features_hook(module, input, output):
        global hook_feature 
        hook_feature = output.view(output.size(0), -1)[0]
        # print("feature shape:{}".format(hook_feature.size()))

def _register_hook(net,layer_name):
    for (name, module) in net.named_modules():
        if name == layer_name:
            module.register_forward_hook(_get_features_hook)

def Load_model(network_name, num_classes, attribute_classes, strategy, pretrained, device):
    if network_name == "ResNet101":
        model = ResNet101(num_classes, attribute_classes, strategy)

    assert os.path.exists(pretrained)
    model_dict = model.state_dict()
    pretrained_param = torch.load(pretrained, map_location=device)

    new_state_dict = OrderedDict()

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
    device = torch.device("cpu")

    # Configuration file
    cfg = Config(args.configuration_file)

    model = Load_model(cfg.BACKBONE, cfg.CLASS_NUM, cfg.ATTRIBUTE_NUM, cfg.STRATEGY, args.Test_model, device)
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
        for xml in tqdm(os.listdir(args.voc2007_anno_path)):
            # Convert XML to Dict format
            xml_path = os.path.join(args.voc2007_anno_path, xml)
            xml_object = open(xml_path, encoding='utf-8')
            xmlStr = xml_object.read()
            xml2dict = xmltodict.parse(xmlStr)

            image_path = os.path.join(args.voc2007_jpeg_path, xml2dict['annotation']['filename'])

            for object in xml2dict['annotation']['object']:
                name = object['name']
                index = np.where(CLASS_NAME==name)

                data = Image.open(image_path)
                data = data.crop((int(object['bndbox']['xmin']),int(object['bndbox']['ymin']),int(object['bndbox']['xmax']),int(object['bndbox']['ymax'])))

                data = transformser(data)

                model(torch.unsqueeze(data, 0).to(device))
                feature = hook_feature
                feature_map[index[0][0]].append(feature.cpu().numpy())
    
    for i in range(20):
        np.save("./feature_representation/VOC_2007/"+CLASS_NAME[i]+'.npy',np.array(feature_map[i]))

def parse_args():
    parser = argparse.ArgumentParser(description='VOC 2008 datasets, attributes prediction')
    parser.add_argument('--configuration-file', type=str,
        default='../configs/Base-ResNet101-B.yaml',
        help='The model configuration file.')
    parser.add_argument('--voc2007-anno-path', type=str,
        default='/exdata/RuoyuChen/few-shot-object-detection/datasets/VOC2007/Annotations/',
        help='The annotations in VOC 2007 datasets.')
    parser.add_argument('--voc2007-jpeg-path', type=str,
        default='/exdata/RuoyuChen/few-shot-object-detection/datasets/VOC2007/JPEGImages/',
        help='The jpeg images in VOC 2007 datasets.')
    parser.add_argument('--gpu-device', type=str, default="1",
                        help='GPU device')
    parser.add_argument('--Test-model', type=str, 
    # default="./checkpoint/backbone-item-epoch-990.pth",
    default="../checkpoint/Save_ckpt_1/backbone-item-epoch-500.pth",
                        help='Model weight for testing.')
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)