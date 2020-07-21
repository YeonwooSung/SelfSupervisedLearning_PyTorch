import torch
import torch.nn as nn
from torch import cat
import torch.nn.init as init

from layer import LRN


class JigsawPuzzleNetwork(nn.module):
    def __init__(self, classes=1000, init_weights=False):
        super(JigsawPuzzleNetwork, self).__init__()


        # ConvNet

        seq_conv = nn.Sequential()
        seq_conv.add_module('conv1_s1', nn.Conv2d(3, 96, kernel_size=11, stride=2, padding=0))
        seq_conv.add_module('relu1_s1', nn.ReLU(inplace=True))
        seq_conv.add_module('pool1_s1', nn.MaxPool2d(kernel_size=3, stride=2))
        seq_conv.add_module('lrn1_s1', LRN(local_size=5, alpha=0.0001, beta=0.75))

        seq_conv.add_module('conv2_s1', nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2))
        seq_conv.add_module('relu2_s1', nn.ReLU(inplace=True))
        seq_conv.add_module('pool2_s1', nn.MaxPool2d(kernel_size=3, stride=2))
        seq_conv.add_module('lrn2_s1', LRN(local_size=5, alpha=0.0001, beta=0.75))

        seq_conv.add_module('conv3_s1', nn.Conv2d(256, 384, kernel_size=3, padding=1))
        seq_conv.add_module('relu3_s1', nn.ReLU(inplace=True))

        seq_conv.add_module('conv4_s1', nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2))
        seq_conv.add_module('relu4_s1', nn.ReLU(inplace=True))

        seq_conv.add_module('conv5_s1', nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2))
        seq_conv.add_module('relu5_s1', nn.ReLU(inplace=True))
        seq_conv.add_module('pool5_s1', nn.MaxPool2d(kernel_size=3, stride=2))

        self.conv = seq_conv


        # FC Layers

        fc6 = nn.Sequential()
        fc6.add_module('fc6_s1', nn.Linear(256*3*3, 1024))
        fc6.add_module('relu6_s1', nn.ReLU(inplace=True))
        fc6.add_module('drop6_s1', nn.Dropout(p=0.5))

        self.fc6 = fc6

        fc7 = nn.Sequential()
        fc7.add_module('fc7', nn.Linear(9*1024, 4096))
        fc7.add_module('relu7', nn.ReLU(inplace=True))
        fc7.add_module('drop7', nn.Dropout(p=0.5))

        self.fc7 = fc7

        self.classifier = nn.Sequential()
        self.classifier.add_module('fc8', nn.Linear(4096, classes))

        if init_weights:
            self.apply(weights_init)

    def load(self, checkpoint):
        model_dict = self.state_dict()

        pretrained_dict = torch.load(checkpoint)
        pretrained_dict = {k: v for k, v in list(pretrained_dict.items()) if k in model_dict and 'fc8' not in k}
        
        model_dict.update(pretrained_dict)
        
        self.load_state_dict(model_dict)
        print([k for k, v in list(pretrained_dict.items())])

    def save(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)

    def forward(self, x):
        B, T, C, H, W = x.size()
        x = x.transpose(0, 1)

        x_list = []
        for i in range(9):
            z = self.conv(x[i])
            z = self.fc6(z.view(B, -1))
            z = z.view([B, 1, -1])
            x_list.append(z)

        x = cat(x_list, 1)
        x = self.fc7(x.view(B, -1))
        x = self.classifier(x)

        return x


def weights_init(model):
    if type(model) in [nn.Conv2d, nn.Linear]:
        nn.init.xavier_normal(model.weight.data)
        nn.init.constant(model.bias.data, 0.1)
