import math
import os
import random
import torch
import itertools
import numpy as np
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler

def cutmix(label_data, unlabel_data, labels, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    lam = min(lam,1-lam)
    bbx1, bby1, bbx2, bby2 = rand_bbox(label_data.size(), lam)
    unlabel_data[:, :, bbx1:bbx2, bby1:bby2] = label_data[:, :, bbx1:bbx2, bby1:bby2]
    return unlabel_data, labels

def rand_bbox(size, lam):
    w = size[2]
    h = size[3]
    cx = np.random.randint(10,90)
    cy = np.random.randint(10,90)
    bbx1 = cx#np.clip(cx - cut_w // 2, 0, w)
    bby1 = cy#np.clip(cy - cut_h // 2, 0, h)
    bbx2 = w-cx#np.clip(cx + cut_w // 2, 0, w)
    bby2 = h-cy#np.clip(cy + cut_h // 2, 0, h)

    return bbx1, bby1, bbx2, bby2

class Estimator():
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()  # intitialization

    def update(self, predictions, targets,argmax=True):
        targets = targets.cpu()
        predictions = predictions.cpu()
        if argmax:
            predictions = torch.tensor([torch.argmax(p) for p in predictions])

        # update metrics
        self.num_samples += len(predictions)
        self.correct += (predictions == targets).sum().item()
        for i, p in enumerate(predictions):
            self.conf_mat[int(targets[i])][int(p.item())] += 1

    def get_accuracy(self, digits=-1):
        acc = self.correct / self.num_samples
        acc = acc if digits == -1 else round(acc, digits)
        return acc

    def get_kappa(self, digits=-1):
        kappa = quadratic_weighted_kappa(self.conf_mat)
        kappa = kappa if digits == -1 else round(kappa, digits)
        return kappa
    
    def get_rec_pre(self, digits=-1):
        cm = self.conf_mat
        recalls = np.zeros(self.num_classes)
        precisions = np.zeros(self.num_classes)
        recalls_fenmu = np.sum(cm, axis=1)
        precisions_fenmu = np.sum(cm, axis=0) # np.diag(cm) /
        for i in range(self.num_classes):
            if recalls_fenmu[i]==0 or precisions_fenmu[i]==0:
                recalls[i]=0
                precisions[i]=0
            else:
                recalls[i]=np.diag(cm)[i]/recalls_fenmu[i]
                precisions[i]=np.diag(cm)[i]/precisions_fenmu[i]
            
        recall = np.mean(recalls)
        precision = np.mean(precisions)
        
        recall = recall if digits == -1 else round(recall, digits)
        precision = precision if digits == -1 else round(precision, digits)

        return recall, precision
    
    def reset(self):
        self.correct = 0
        self.num_samples = 0
        self.conf_mat = np.zeros((self.num_classes, self.num_classes), dtype=int)

            
def quadratic_weighted_kappa(conf_mat):
    assert conf_mat.shape[0] == conf_mat.shape[1]
    cate_num = conf_mat.shape[0]
    weighted_matrix = np.zeros((cate_num, cate_num))
    for i in range(cate_num):
        for j in range(cate_num):
            weighted_matrix[i][j] = 1 - float(((i - j)**2) / ((cate_num - 1)**2))
    ground_truth_count = np.sum(conf_mat, axis=1)
    pred_count = np.sum(conf_mat, axis=0)
    expected_matrix = np.outer(ground_truth_count, pred_count)

    conf_mat = conf_mat / conf_mat.sum()
    expected_matrix = expected_matrix / expected_matrix.sum()

    observed = (conf_mat * weighted_matrix).sum()
    expected = (expected_matrix * weighted_matrix).sum()
    return (observed - expected) / (1 - expected)

def adjust_learning_rate(learning_rate,epochs, optimizer, epoch):
    warmup_epochs = 5
    if epoch < warmup_epochs:
        lr = learning_rate * epoch / warmup_epochs
    else:
        lr = learning_rate * 0.5 * (1. + math.cos(math.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
    
def print_dataset_info(datasets):
    train_dataset,val_dataset, test_dataset = datasets
    print('Dataset Loaded.')
    print('Categories:\t{}'.format(len(train_dataset.classes)))
    print('Training:\t{}'.format(len(train_dataset)))
    print('Validation:\t{}'.format(len(val_dataset)))
    print('Test:\t\t{}'.format(len(test_dataset)))

def print_class_info(dataset):
    class_to_idx = dataset.class_to_idx
    class_counts = {}
    for _, label in dataset.samples:
        class_name = dataset.classes[label]
        if class_name in class_counts:
            class_counts[class_name] += 1
        else:
            class_counts[class_name] = 1
    print(class_counts)
    
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
class TwoStreamBatchSampler(Sampler):
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self): 
        primary_iter = iterate_once(self.primary_indices) 
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (primary_batch + secondary_batch for (primary_batch, secondary_batch) 
                in zip(grouper(primary_iter, self.primary_batch_size),grouper(secondary_iter, self.secondary_batch_size)))

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

def iterate_once(iterable):
    return np.random.permutation(iterable) 

def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())

def grouper(iterable, n):
    args = [iter(iterable)] * n
    return zip(*args)

def get_current_consistency_weight(consistency,current_step, total_steps):
    phase = 1.0 - current_step / total_steps
    final_consistency = consistency* np.exp(-5.0 * phase * phase)
    return final_consistency 

def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
        
def evaluate(learning_rate,batch_size, model, checkpoint, test_dataset, estimator, test = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = torch.load(checkpoint)
    model.load_state_dict(weights, strict=True)
    test_loader = DataLoader(test_dataset,batch_size=batch_size,num_workers=4,shuffle=False,pin_memory=True)

    eval(model, test_loader, estimator, device)
    Acc, Kappa = estimator.get_accuracy(4), estimator.get_kappa(4)
    recall,precision = estimator.get_rec_pre(4)
    print('Test acc: {}'.format(Acc),' kappa: {}'.format(Kappa),' Recall: {}'.format(recall),' Precision: {}'.format(precision))
    print('Confusion Matrix:')
    print(estimator.conf_mat)
    if test:
        with open("./eval.txt","a") as f:
            txt = 'test acc:'+str(Acc)+' kappa:'+str(Kappa)+' recall:'+str(recall)+' precision:'+str(precision) +' LR = '+str(learning_rate)
            f.write(txt+'\n') 

def eval(model, dataloader, estimator, device):
    model.eval()
    torch.set_grad_enabled(False)
    estimator.reset()
    for test_data in dataloader:
        X, y = test_data
        X, y = X.to(device), y.to(device).float()
        pred = model(X)
        if isinstance(pred, tuple):
            y_pred,_,_,_,_= model(X)
            estimator.update(y_pred, y)
        else:
            y_pred = model(X)
            estimator.update(y_pred, y)
    model.train()
    torch.set_grad_enabled(True)
    
def mb_sup_loss(logits_ova, label):
    batch_size = logits_ova.size(0)
    logits_ova = logits_ova.view(batch_size, 2, -1)
    num_classes = logits_ova.size(2)
    probs_ova = F.softmax(logits_ova, 1)
    label_s_sp = torch.zeros((batch_size, num_classes)).long().to(label.device)
    label_range = torch.arange(0, batch_size).long().to(label.device)
    label_s_sp[label_range[label < num_classes], label[label < num_classes]] = 1
    label_sp_neg = 1 - label_s_sp
    #print(label_s_sp)
    open_loss = torch.mean(torch.sum(-torch.log(probs_ova[:, 1, :] + 1e-8) * label_s_sp, 1))
    open_loss_neg = torch.mean(torch.max(-torch.log(probs_ova[:, 0, :] + 1e-8) * label_sp_neg, 1)[0])
    l_ova_sup = open_loss_neg + open_loss
    return l_ova_sup    
'''
def OODevaluate(model, checkpoint, test_loader, estimator):
    torch.set_printoptions(precision=6, sci_mode=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dict = torch.load(checkpoint)
    model.load_state_dict(model_dict)
    model.eval()
    torch.set_grad_enabled(False)
    estimator.reset()
    for X, y in test_loader:
        X, y = X.to(device), y
        y_pred,_,unlabel_features,queue,mb_out= model(X)
        #out_soft = torch.softmax(y_pred, dim=1)
        #print(out_soft)
        y_pred = torch.tensor([torch.argmax(p) for p in y_pred])
    
        #unlabel_features = nn.functional.normalize(unlabel_features, dim=1)
        #unlabel_features = unlabel_features.unsqueeze(1) #B*1*C
        #queue = queue.unsqueeze(0)#1*K*C
        #similarity = F.cosine_similarity(unlabel_features, queue, dim=2) #B*K
        
        similarity = (unlabel_features @ queue.transpose(-2, -1)* 256 ** -0.5)
        #similarity =  torch.mean(similarity, dim=1)
        similarity = nn.Softmax(dim=-1)(similarity)
        
        max_values, max_indices = torch.max(similarity, dim=-1)
        index = max_values>0.80
        y_pred[~index] = estimator.num_classes-1
        estimator.update(y_pred, y,argmax=False) 
        
    Acc,Kappa = estimator.get_accuracy(4),estimator.get_kappa(4)
    recall,precision = estimator.get_rec_pre(4)
    print('Test acc: {}'.format(Acc),' kappa: {}'.format(Kappa),' Recall: {}'.format(recall),' Precision: {}'.format(precision))
    print('Confusion Matrix:')
    print(estimator.conf_mat)
'''
def OODevaluate(model, checkpoint, test_loader, estimator):
    torch.set_printoptions(precision=6, sci_mode=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dict = torch.load(checkpoint)
    model.load_state_dict(model_dict)
    model.eval()
    torch.set_grad_enabled(False)
    estimator.reset()
    for X, y in test_loader:
        X, y = X.to(device), y
        y_pred,_,unlabel_features,queue,mb_out= model(X)
        
        out_soft = torch.softmax(y_pred, dim=1)
        mb_out = torch.softmax(mb_out.view(mb_out.size(0), 2, -1), 1)
        mb_out = mb_out[:,1,:]
        
        y_pred = torch.tensor([torch.argmax(p) for p in y_pred])
        similarity = (unlabel_features @ queue.transpose(-2, -1)* 256 ** -0.5)
        similarity = nn.Softmax(dim=-1)(similarity)
        similarity = similarity * mb_out
        max_values, max_indices = torch.max(similarity, dim=-1)
        
        index = max_values>0.4
        y_pred[~index] = estimator.num_classes-1
        estimator.update(y_pred, y,argmax=False) 
        
    Acc,Kappa = estimator.get_accuracy(4),estimator.get_kappa(4)
    recall,precision = estimator.get_rec_pre(4)
    print('Test acc: {}'.format(Acc),' kappa: {}'.format(Kappa),' Recall: {}'.format(recall),' Precision: {}'.format(precision))
    print('Confusion Matrix:')
    print(estimator.conf_mat)
