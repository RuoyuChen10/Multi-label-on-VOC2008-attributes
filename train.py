# -*- coding: utf-8 -*-  

"""
Created on 2021/07/14

@author: Ruoyu Chen
"""

import argparse
import os
import numpy as np
import time

import torch
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

from config import Config
from Logging import Logger
from dataset import Dataset

from prettytable import PrettyTable
from models.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from Loss import MultiClassLoss

def mkdir(name):
    '''
    Create folder
    '''
    isExists=os.path.exists(name)
    if not isExists:
        os.makedirs(name)
    return 0

def define_backbone(network_name, num_classes, attribute_classes, strategy, pretrained=None):
    if network_name == "ResNet50":
        model = ResNet50(num_classes, attribute_classes, strategy)
    elif network_name == "ResNet101":
        model = ResNet101(num_classes, attribute_classes, strategy)

    if os.path.exists(pretrained):      # Load pretrained model
        model_dict = model.state_dict()
        pretrained_param = torch.load(pretrained)
        pretrained_dict = {k: v for k, v in pretrained_param.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        
        logger.write("\033[32m{} ".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())) + 
                     "\033[0mLoad model" + 
                     "\033[34m {} ".format(network_name) + 
                     "\033[0mfrom pretrained" + 
                     "\033[34m {}".format(pretrained))
    else:               # Initialize from zero
        logger.write("\033[0mChoose network" + 
                     "\033[34m {} ".format(network_name) + 
                     "\033[0mas backbone.")
    return model

def define_Loss_function(loss_name, pos_loss_weight=None, weight = None):
    if loss_name == "Multi":
        Loss_function = MultiClassLoss()
    elif loss_name == "BCELoss":
        Loss_function = nn.BCEWithLogitsLoss(weight=weight, pos_weight=pos_loss_weight)
    return Loss_function

def define_optimizer(model, optimizer_name, learning_rate):
    if optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr = learning_rate, weight_decay = 0.01)
    elif optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr = learning_rate, weight_decay = 0.01)
    return optimizer

def TPFN(model, validation_loader, attribute_num, attribute_name, thresh, device):
    """
    Compute the TP, FN, FP, TN
    Precision = TP / (TP + FP)
    Recell = TP / (TP + FN)
    """
    model.eval()

    corrects = np.zeros((attribute_num, 4))   # [Batch, (TN,FN,FP,TP)]
    with torch.no_grad():
        for i, (data,labels) in enumerate(validation_loader):
            data = data.to(device)
            labels = labels.to(device)

            outputs = sigmoid(model(data))

            ii = 0
            for output, label in zip(outputs.t(), labels.t()):
                # output: Torch_size(batch)
                # label: Torch_size(batch)
                output_label = (output>thresh).int()
                # if ii ==1:
                #     print(output)

                results = output_label * 2 + label
                TN_n = len(torch.where(results==0)[0])
                FN_n = len(torch.where(results==1)[0])
                FP_n = len(torch.where(results==2)[0])
                TP_n = len(torch.where(results==3)[0])

                assert len(results) == TN_n + FN_n + FP_n + TP_n

                corrects[ii][0] += TN_n; corrects[ii][1] += FN_n
                corrects[ii][2] += FP_n; corrects[ii][3] += TP_n
                ii += 1
    
    table = PrettyTable(["Attribute Name", "TP", "FN", "FP", "TN","ACC"])
    
    for i in range(attribute_num):
        table.add_row([str(i)+". "+attribute_name[i],
                       corrects[i][3],  # TP
                       corrects[i][1],  # FN
                       corrects[i][2],  # FP
                       corrects[i][0],  # TN
                       "%.4f"%((corrects[i][3]+corrects[i][0])/(corrects[i][0]+corrects[i][1]+corrects[i][2]+corrects[i][3]))    # ACC
                       ]) 
    print(table)

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device
    device = torch.device("cuda")

    # Configuration file
    global cfg
    cfg = Config(args.configuration_file)

    # model save path
    model_save_path = os.path.join(cfg.CKPT_SAVE_PATH, 
        time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()))
    mkdir(model_save_path)

    # logger
    global logger
    logger = Logger(os.path.join(model_save_path,"logging.log"))

    # Dataloader
    train_dataset = Dataset(dataset_root=cfg.DATASET_ROOT,dataset_list=cfg.DATASET_LIST_TRAIN,class_name=cfg.CLASS_NAME, strategy=cfg.STRATEGY, data_type="train")
    train_loader = DataLoader(train_dataset,batch_size=cfg.BATCH_SIZE,shuffle=True)

    validation_dataset = Dataset(dataset_root=cfg.DATASET_ROOT,dataset_list=cfg.DATASET_LIST_TEST,class_name=cfg.CLASS_NAME, strategy=cfg.STRATEGY, data_type="test")
    validation_loader = DataLoader(validation_dataset,batch_size=cfg.BATCH_SIZE,shuffle=False)

    model = define_backbone(cfg.BACKBONE, cfg.CLASS_NUM, cfg.ATTRIBUTE_NUM, cfg.STRATEGY ,cfg.PRETRAINED)

    # GPU
    if torch.cuda.is_available():
        model = model.cuda()
    # Multi GPU
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # Loss
    # weight = torch.Tensor([cfg.ATTR_LOSS_WEIGHT for i in range(cfg.BATCH_SIZE)]).to(device)
    # weight = torch.Tensor(cfg.ATTR_LOSS_WEIGHT).to(device)
    Loss_function = define_Loss_function(cfg.LOSS_FUNCTION, torch.Tensor(cfg.POS_LOSS_WEIGHT).to(device),weight=None)

    # optimizer
    optimizer = define_optimizer(model, cfg.OPTIMIZER, cfg.LEARNING_RATE)

    scheduler = StepLR(optimizer, step_size=40, gamma=0.1)

    for i in range(1,cfg.EPOCH+1):
        scheduler.step()

        model.train()
        
        for ii, (data,label) in enumerate(train_loader):
            data = data.to(device)
            label = label.to(device)

            output = model(data)
            loss = Loss_function(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iters = i* len(train_loader) + ii

            if iters % 10 == 0:
                logger.write(
                    "\033[32m{} ".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())) + 
                    "\033[0mtrain epoch " + 
                    "\033[34m{} ".format(i) +
                    "\033[0miter " + 
                    "\033[34m{} ".format(ii) + 
                    "\033[0mloss " + 
                    "\033[34m{}.".format(loss.item())
                )

        if i % 10 == 0:
            TPFN(model, validation_loader, cfg.ATTRIBUTE_NUM, cfg.ATTRIBUTE_NAME, 0.5, device)
            torch.save(model.state_dict(), os.path.join(model_save_path,"backbone-item-epoch-"+str(i)+'.pth'))


def parse_args():
    parser = argparse.ArgumentParser(description='VOC 2008 datasets, attributes prediction')
    parser.add_argument('--configuration-file', type=str,
        default='./configs/Base-ResNet101-B.yaml',
        help='The model configuration file.')
    parser.add_argument('--gpu-device', type=str, default="2,3",
                        help='GPU device')
    args = parser.parse_args()

    return args
    
if __name__ == "__main__":
    sigmoid = nn.Sigmoid()
    sigmoid.to(torch.device("cuda"))
    args = parse_args()
    main(args)