# -*- coding: utf-8 -*-
"""
@original_author: Biagio Brattoli
@modified_by: Yeonwoo Sung
"""

import torch
import torch.nn as nn
from torch import cat
import torch.nn.init as init

import sys
sys.path.append('Utils')
from Layers import LRN

class JigsawResnet(nn.Module):

    def __init__(self, classes=1000):
        super(JigsawResnet, self).__init__()

        self.conv1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2),
                nn.ReLU(inplace=True)
        )


        self.fc6 = nn.Sequential()
        self.fc6.add_module('fc6_s1',nn.Linear(256*3*3, 1024))
        self.fc6.add_module('relu6_s1',nn.ReLU(inplace=True))
        self.fc6.add_module('drop6_s1',nn.Dropout(p=0.5))

        self.fc7 = nn.Sequential()
        self.fc7.add_module('fc7',nn.Linear(9*1024,4096))
        self.fc7.add_module('relu7',nn.ReLU(inplace=True))
        self.fc7.add_module('drop7',nn.Dropout(p=0.5))

        self.classifier = nn.Sequential()
        self.classifier.add_module('fc8',nn.Linear(4096, classes))
        

    def load(self,checkpoint):
        model_dict = self.state_dict()
        pretrained_dict = torch.load(checkpoint)
        pretrained_dict = {k: v for k, v in list(pretrained_dict.items()) if k in model_dict and 'fc8' not in k}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        print([k for k, v in list(pretrained_dict.items())])

    def save(self,checkpoint):
        torch.save(self.state_dict(), checkpoint)
    
    def forward(self, x):
        B,T,C,H,W = x.size()
        x = x.transpose(0,1)

        x_list = []
        for i in range(9):
            z = self.conv(x[i])
            z = self.fc6(z.view(B,-1))
            z = z.view([B,1,-1])
            x_list.append(z)

        x = torch.cat(x_list,1)
        x = self.fc7(x.view(B,-1))
        x = self.classifier(x)

        return x
