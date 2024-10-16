import torch
import numpy as np
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from newmodel import *
#from openvit import *

from typing import Type, Any, Callable, Union, List, Optional, cast, Tuple

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):
    expansion: int = 4
    def __init__(self,inplanes: int,planes: int,stride: int = 1,downsample: Optional[nn.Module] = None,
                 groups: int = 1,base_width: int = 64,dilation: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self,block, layers,num_classes,zero_init_residual=False,groups = 1,width_per_group = 64): 
        super(ResNet, self).__init__()
        
        norm_layer = nn.BatchNorm2d
        self._norm_layer = nn.BatchNorm2d
        self.inplanes = 64
        self.dilation = 1
        replace_stride_with_dilation = [False, False, False]
            
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)
        
    def _make_layer(self, block, planes, blocks, stride = 1, dilate = False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),norm_layer(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
       
        return x

class OpenResNet(nn.Module):
    def __init__(self,block, layers,num_classes,zero_init_residual=False,groups = 1,width_per_group = 64): 
        super(OpenResNet, self).__init__()
        
        norm_layer = nn.BatchNorm2d
        self._norm_layer = nn.BatchNorm2d
        self.inplanes = 64
        self.dilation = 1
        replace_stride_with_dilation = [False, False, False]
            
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        
        self.dimension =256
        self.num_classes = num_classes
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.transform = nn.Linear(512 * 4, self.dimension*self.num_classes)
        self.fc = nn.Linear(self.dimension*self.num_classes, self.num_classes)
        
        self.prompt = nn.Parameter(torch.randn(self.num_classes, self.dimension))
        self.scale = self.dimension ** -0.5
        self.softmax = nn.Softmax(dim=-1)

    def _make_layer(self, block, planes, blocks, stride = 1, dilate = False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),norm_layer(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor,labels=None, update=None) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        features = x.view(x.size(0), -1)
        
        features = self.transform(features)
        out_features = self.relu(features)
        B,C = out_features.shape
        features = out_features.reshape(B, self.num_classes, self.dimension)
        
        prompt_attn = (features @ self.prompt.transpose(-2, -1)* self.scale)
        prompt_attn = torch.mean(prompt_attn, dim=1)

        out = self.fc(out_features)
        return out,prompt_attn,features,self.prompt

def OpenResnet50(img_size,num_classes) :
    model = OpenResNet(Bottleneck, [3, 4, 6, 3],num_classes = num_classes)
    res_dict = torch.load('./resnet50.pth',map_location=torch.device('cpu'))
    model_dict = model.state_dict()
    matched_dict = {k: v for k, v in res_dict.items() if k in model_dict and v.shape==model_dict[k].shape}
    print('matched_dict:',len(matched_dict))
    model_dict.update(matched_dict)
    model.load_state_dict(model_dict)
    return model
    
def resnet50(img_size,num_classes) :
    model = ResNet(Bottleneck, [3, 4, 6, 3],num_classes = num_classes)
    res_dict = torch.load('./resnet50.pth',map_location=torch.device('cpu'))
    model_dict = model.state_dict()
    matched_dict = {k: v for k, v in res_dict.items() if k in model_dict and v.shape==model_dict[k].shape}
    print('matched_dict:',len(matched_dict))
    model_dict.update(matched_dict)
    model.load_state_dict(model_dict)
    return model

def create_resnet_model(image_size, num_classes,ema):
    model = resnet50(image_size,num_classes)
    if ema:
        for param in model.parameters():
            param.requires_grad = False
    return model

def create_open_resnet_model(image_size, num_classes,ema):
    #model = OpenResnet50(image_size,num_classes)
    model = ViT_AdptFormer_Model(image_size,num_classes)
    if ema:
        for param in model.parameters():
            param.requires_grad = False
    return model