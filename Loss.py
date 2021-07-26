# -*- coding: utf-8 -*-  

"""
Created on 2021/07/15

@author: Ruoyu Chen
"""
import torch
import torch.nn as nn 

class MultiClassLoss(nn.Module):
    def __init__(self):
        super(MultiClassLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()  # Input (N,C), Target (C)
    def forward(self, outs, labels):
        """
        outs: List[Torch_size(batch,33)]
        labels: Torch_size(batch, attributes)
        """
        loss = 0
        #loss_information = []
        for out,label in zip(outs,labels.t()):
            # out: Torch_size(batch,33)
            # label: Torch_size(batch)
            criterion_loss = self.criterion(out, label)
            loss += criterion_loss
            #loss_information.append(criterion_loss.data.item())
        return loss#,loss_information