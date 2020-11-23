from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
import numpy as np
import torch.optim as optim
import os
import time
import imageio
import datetime
try:
    import cPickle
except:
    import _pickle as cPickle


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('Total:', total_num, 'Trainable:', trainable_num)

    return total_num, trainable_num

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride):
        super(InvertedResidual, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        #out = channel_shuffle(out, 2)

        return out

class Net1(nn.Module):
    def __init__(self, stages_out_channels, stages_repeats, num_classes=2):
        super(Net1, self).__init__()

        self._stage_out_channels = stages_out_channels

        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stage_names = ['conv{}'.format(i) for i in range(2,len(stages_out_channels))]
        for name, repeats, output_channels in zip(
                stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [InvertedResidual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(InvertedResidual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        output_channels = self._stage_out_channels[-1]
        """last_conv_name = 'conv%i'%len(stages_out_channels)
        seq = [nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)]
        setattr(self, last_conv_name, nn.Sequential(*seq))"""
        input_channels = output_channels
        """inter_channels = 256
        output_channels = 128
        self.mlp = nn.Sequential(nn.Dropout(0.5), nn.Linear(input_channels,output_channels))"""
        self.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(input_channels, num_classes))

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        f = x.mean([2,3])
        #f1 = self.mlp(f)
        y = self.fc(f)
        return f, y

class Net2(nn.Module):
    def __init__(self, stages_out_channels, stages_repeats, num_classes=2):
        super(Net2, self).__init__()

        self._stage_out_channels = stages_out_channels

        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stage_names = ['conv{}'.format(i) for i in range(2,len(stages_out_channels))]
        for name, repeats, output_channels in zip(
                stage_names, stages_repeats, self._stage_out_channels[1:]):
            """seq = [InvertedResidual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(InvertedResidual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))"""
            seq = [nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)]
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        output_channels = self._stage_out_channels[-1]
        last_conv_name = 'conv%i'%len(stages_out_channels)
        seq = [nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)]
        setattr(self, last_conv_name, nn.Sequential(*seq))
        input_channels = output_channels
        """inter_channels = 256
        output_channels = 128
        self.mlp = nn.Sequential(nn.Dropout(0.5), nn.Linear(input_channels,output_channels))"""
        self.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(input_channels, num_classes))

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        f = x.mean([2,3])
        #f1 = self.mlp(f)
        y = self.fc(f)
        return f, y

if __name__ == '__main__':
    m = models.resnet18(pretrained=True).cuda()
    m1 = Net2([24, 48, 96, 192, 1024],[1,1,1],1)
    get_parameter_number(m1)
    #print(m1)
    mm = Net1([24, 48, 96, 192, 1024],[1,1,1],1)
    """f2 = open('net.txt', 'w')
    f2.write(str(mm))"""
    get_parameter_number(mm)
    x = torch.randn(1, 3, 300, 300)
    print(mm(x)[1].size())
    """f1 = open('shufflenetv2.txt','w')
    f1.write(str(m1))
    """
    #print(mm)