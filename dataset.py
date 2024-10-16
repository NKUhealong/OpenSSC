import os
import torch
import numpy as np
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset, DataLoader
from typing import Type, Any, Callable, Union, List, Optional, cast, Tuple

from utils import *

def lab_conv(knownclass, label):
    knownclass = sorted(knownclass)
    label_convert = torch.zeros(len(label), dtype=int)
    for j in range(len(label)):
        for i in range(len(knownclass)):

            if label[j] == knownclass[i]:
                label_convert[j] = int(knownclass.index(knownclass[i]))
                break
            else:
                label_convert[j] = int(len(knownclass))     
    return label_convert

##  for open set
def get_dataset_splits(image_size,batch_size, dataset, selected_known_classes, ratio, is_open):
    augmentations = [transforms.RandomHorizontalFlip(p=0.5),transforms.RandomVerticalFlip(p=0.5),
                     #transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.1,hue=0.1),
                     transforms.RandomRotation(degrees=(-180, 180)),transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                     transforms.Resize(image_size),transforms.ToTensor()]
    train_transform = transforms.Compose(augmentations)
    test_transform = transforms.Compose([transforms.Resize(image_size),transforms.ToTensor()])
    
    if dataset == 'skin':
        train_path = os.path.join('./data/ISIC_2018/' , 'train')
        test_path = os.path.join('./data/ISIC_2018/' , 'test')
        train_class_num = np.array([327,514,1099,115,1113,6705,142])
        test_class_num = np.array([43,93,217,44,171,909,35])
    elif dataset == 'pathology':
        train_path = os.path.join('./data/NCT-CRC-HE-100K/' , 'train')
        test_path = os.path.join('./data/NCT-CRC-HE-100K/' , 'test')
        train_class_num =  np.array([7285,7396,8058,8090,6227,9476,6134,7312,10022])
        test_class_num =  np.array([3122,3170,3454,3467,2669,4060,2629,3134,4295])
    elif dataset == 'DDR':
        train_path = os.path.join('./data/DDRCls/' , 'train')
        test_path = os.path.join('./data/DDRCls/' , 'test')
        train_class_num =  np.array([2992,1638,2238,613,2370,575])
        test_class_num  =  np.array([1880,189,1344,71,275,346])

    train_dataset = datasets.ImageFolder(train_path, train_transform, loader=pil_loader)
    train_cumu_sum = np.cumsum(train_class_num).astype(int)
    train_cumu_sum = np.insert(train_cumu_sum, 0, 0, axis=0)
    train_label_num = (train_class_num*ratio).astype(int)
    label_samples,unlabel_samples = [], []
    for i in range(len(train_class_num)):
        if i in selected_known_classes:
            label_samples = label_samples+train_dataset.samples[train_cumu_sum[i]:train_cumu_sum[i]+train_label_num[i]]
            unlabel_samples = unlabel_samples+train_dataset.samples[train_cumu_sum[i]+train_label_num[i]:train_cumu_sum[i+1]]
        elif is_open:
            unlabel_samples = unlabel_samples+train_dataset.samples[train_cumu_sum[i]:train_cumu_sum[i+1]]
    train_dataset.samples = label_samples + unlabel_samples
    labeled_bs = int(batch_size/2)
    print("=> Total samples: {}, labeled is: {}".format(len(train_dataset), len(label_samples)))
    labeled_idxs = list(range(0, len(label_samples)))*int(len(unlabel_samples)/len(label_samples))
    unlabeled_idxs = list(range(len(label_samples), len(train_dataset)))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size-labeled_bs)

    ID_test_dataset = datasets.ImageFolder(test_path, test_transform, loader=pil_loader)
    test_cumu_sum = np.cumsum(test_class_num).astype(int)
    test_cumu_sum = np.insert(test_cumu_sum, 0, 0, axis=0)
    test_samples = []
    for i in range(len(test_class_num)):
        if i in selected_known_classes:
            test_samples = test_samples+ID_test_dataset.samples[test_cumu_sum[i]:test_cumu_sum[i+1]]
    ID_test_dataset.samples = test_samples

    train_targets,test_targets = [], []
    for x, y in train_dataset.samples:
        train_targets = train_targets+[y]
    for x, y in ID_test_dataset.samples:
        test_targets = test_targets+[y]

    train_trans_labels = lab_conv(selected_known_classes, train_targets)
    test_trans_labels = lab_conv(selected_known_classes, test_targets)

    for i in range(len(train_trans_labels)):
        x,y = train_dataset.samples[i]
        train_dataset.samples[i]=(x,train_trans_labels[i])
    for i in range(len(test_trans_labels)):
        x,y = ID_test_dataset.samples[i]
        ID_test_dataset.samples[i]=(x,test_trans_labels[i])
        
    OOD_test_dataset = datasets.ImageFolder(test_path, test_transform, loader=pil_loader)
    OOD_trans_labels = lab_conv(selected_known_classes, OOD_test_dataset.targets)
    OOD_samples = []
    for i in range(len(OOD_trans_labels)):
        x,y = OOD_test_dataset.samples[i]
        OOD_test_dataset.samples[i]=(x,OOD_trans_labels[i])
    print_class_info(train_dataset)
    print_class_info(ID_test_dataset)
    print_class_info(OOD_test_dataset)
    
    return batch_sampler,train_dataset, ID_test_dataset, OOD_test_dataset
 
##  for close set
def generate_dataset(dataset,input_size,ratio = 0.6):
    augmentations = [transforms.RandomHorizontalFlip(p=0.5),transforms.RandomVerticalFlip(p=0.5),
                     transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.1,hue=0.1),
                     transforms.RandomRotation(degrees=(-180, 180)),
                     transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                     transforms.Resize((input_size, input_size)),transforms.ToTensor()]
    train_transform = transforms.Compose(augmentations)
    test_transform = transforms.Compose([transforms.Resize((input_size, input_size)),transforms.ToTensor()])
    
    if dataset == 'skin':
        train_path = os.path.join('./data/ISIC_2018/' , 'train')
        test_path = os.path.join('./data/ISIC_2018/' , 'test')
        train_class_num = np.array([327,514,1099,115,1113,6705,142])        
    elif dataset == 'pathology':
        train_path = os.path.join('./data/NCT-CRC-HE-100K/' , 'train')
        test_path = os.path.join('./data/NCT-CRC-HE-100K/' , 'test')
        train_class_num =  np.array([7285,7396,8058,8090,6227,9476,6134,7312,10022])
    elif dataset == 'DDR':
        train_path = os.path.join('./data/DDRCls/' , 'train')
        test_path = os.path.join('./data/DDRCls/' , 'test')
        train_class_num =  np.array([2992,1638,2238,613,2370,575])
    train_dataset = datasets.ImageFolder(train_path, train_transform, loader=pil_loader)
    print_class_info(train_dataset)
    cumu_sum = np.cumsum(train_class_num).astype(int)
    label_num = (train_class_num*ratio).astype(int)
    samples = train_dataset.samples[0:label_num[0]]
    for i in range(len(train_class_num)):
        samples = samples+train_dataset.samples[cumu_sum[i]:cumu_sum[i]+label_num[i]]
    train_dataset.samples = samples
    
    test_dataset = datasets.ImageFolder(test_path, test_transform, loader=pil_loader)
    print_class_info(test_dataset)
    dataset = train_dataset,test_dataset,test_dataset
    print_dataset_info(dataset)
    return dataset